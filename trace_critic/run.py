"""End-to-end experiment: contracts -> rollout labels -> critic (AdamW vs Muon) -> controller eval -> TRACE.

Usage: uv run python -m trace_critic.run [--executor sim|nemotron]
"""

import argparse
import json
import math
import random
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import critic as C
from .env import RUNS, VERIFIER_VERSION, get_contract, warm_cache
from .executor import NemotronExecutor, SimulatedExecutor
from .search import FEAT_DIM, features, graph_search, make_node, ralph, rule_scorer
from .tasks import TASKS, TEST, TRAIN, VALID

N_TRAJ = 25         # source trajectories per task
SRC_STEPS = 10      # max steps per source trajectory
K_CONT = 6          # continuations per state
BUDGETS = (4, 8, 16)
EVAL_SEEDS = 60     # episodes per held-out task per controller
EVAL_BUDGET = 16    # executor calls per episode (all controllers)
POOL = 1            # parallel episodes (1 keeps the simulated run byte-identical)
OUT = RUNS          # output directory
# Smaller config for a real LLM executor (every proposal is an API call).
NEMOTRON_CFG = dict(N_TRAJ=4, SRC_STEPS=8, K_CONT=3, BUDGETS=(4, 8), EVAL_SEEDS=10, POOL=24)


def seed_of(*xs):
    return zlib.crc32("|".join(map(str, xs)).encode())


def pmap(fn, items):
    if POOL <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(POOL) as ex:
        return list(ex.map(fn, items))


def collect(task_name, executor, split):
    from .search import verified

    def source(t):
        rng = random.Random(seed_of("src", task_name, t))
        root = make_node(task_name, TASKS[task_name].root)
        log = []
        ralph(root, executor, SRC_STEPS, rng, log)
        states = [root] + [e["node"] for e in log if e["decision"] == "CONTINUE"]
        uniq = {}
        for s in states:
            uniq.setdefault((s.artifact, s.history), s)
        return [(t, si, s) for si, s in enumerate(uniq.values()) if not verified(s)]  # terminal: no label needed

    jobs = [j for js in pmap(source, range(N_TRAJ)) for j in js]

    def label(job):
        t, si, s = job
        B = BUDGETS[seed_of(task_name, t, si) % len(BUDGETS)]
        succ = valid = 0
        for k in range(K_CONT):
            try:
                ok, *_ = ralph(s, executor, B, random.Random(seed_of("cont", task_name, t, si, k)))
                succ += ok
                valid += 1
            except Exception as e:  # invalid_run (verifier or infra failure): excluded from label, reported
                print("invalid_run:", type(e).__name__, str(e)[:120])
        if valid == 0:
            return None
        return (dict(task_id=task_name, split=split, trajectory_id=f"run-{t:02d}", state_id=f"state-{si:02d}",
                             artifact_hash=s.artifact, prefix_len=len(s.history), observed_checks=s.observed,
                             policy_id=executor.policy_id, continuation_budget={"executor_calls": B},
                             labels={"kind": "rollout_outcome", "successes": succ, "valid_rollouts": valid,
                                     "invalid_runs": K_CONT - valid},
                             verifier_version=VERIFIER_VERSION, contract_version="c1",
                             _x=features(s, B).tolist(), _q=succ / valid))

    return [r for r in pmap(label, jobs) if r is not None]


def XY(rows):
    return (np.array([r["_x"] for r in rows], dtype=np.float32),
            np.array([r["_q"] for r in rows], dtype=np.float32))


def critic_metrics(pred, rows):
    q = np.array([r["_q"] for r in rows])
    out = {"n_states": len(rows), "mse": float(((pred - q) ** 2).mean())}
    # pairwise ranking among states of the same task & budget with different labels
    agree = tot = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a["task_id"] == b["task_id"] and a["continuation_budget"] == b["continuation_budget"] and abs(q[i] - q[j]) > 0.2:
                tot += 1
                agree += (pred[i] - pred[j]) * (q[i] - q[j]) > 0
    out["pairwise_rank_acc"] = agree / tot if tot else None
    out["pairs"] = tot
    # same visible checks (all visible passing): can the critic see past the visible test status?
    m = np.array([all(r["observed_checks"].values()) for r in rows])
    out["n_all_visible_pass"] = int(m.sum())
    out["mse_all_visible_pass"] = float(((pred[m] - q[m]) ** 2).mean()) if m.any() else None
    bins = []
    for lo in np.linspace(0, 0.8, 5):
        sel = (pred >= lo) & (pred < lo + 0.2 + (1e-9 if lo >= 0.8 else 0))
        if sel.any():
            bins.append({"bin": f"{lo:.1f}-{lo + 0.2:.1f}", "n": int(sel.sum()),
                         "mean_pred": float(pred[sel].mean()), "mean_label": float(q[sel].mean())})
    out["calibration"] = bins
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def evaluate(executor, critics):
    ctrls = {"ralph": None, "graph_rules": rule_scorer}
    for name, m in critics.items():
        ctrls[f"graph_learned_{name}"] = (lambda mm: lambda n, b: float(C.predict(mm, features(n, b)[None][:, mm.cols])[0]))(m)
    res, episodes = {}, {}
    for cname, scorer in ctrls.items():
        agg = dict(episodes=0, success=0, calls=0, calls_success=[], vcalls=0, verify_fail=0, loops=0, gaps=0)

        def run_ep(ts, scorer=scorer):
            task, s = ts
            if True:
                rng = random.Random(seed_of("eval", task, s))
                root = make_node(task, TASKS[task].root)
                if scorer is None:
                    log = []
                    ok, calls, vcalls, _ = ralph(root, executor, EVAL_BUDGET, rng, log)
                    seen, loops = {root.artifact}, 0
                    for e in log:
                        if e["decision"] == "CONTINUE":
                            loops += e["node"].artifact in seen
                            seen.add(e["node"].artifact)
                    evs = [dict(id=root.id, parent=None, decision="ROOT", score=None, calls=0, action=None,
                                vis=sum(root.observed.values()) / len(root.observed), verified=None,
                                variant=root.variant)]
                    n_calls = 0
                    for e in log:
                        n_calls += e["decision"] == "CONTINUE"
                        evs.append(dict(id=e["node"].id, parent=e["parent"].id, decision=e["decision"], score=None,
                                        calls=n_calls, variant=e["node"].variant,
                                        vis=sum(e["node"].observed.values()) / len(e["node"].observed),
                                        verified=(True if e["decision"] == "VERIFIED" else False if e["decision"] == "FLAG_GAP" else None),
                                        action=e["node"].last_action))
                    if not ok:
                        evs.append(dict(id=None, parent=None, decision="STOP_UNRESOLVED", score=None, calls=calls,
                                        vis=None, verified=False, action=None, variant=None))
                    r = dict(success=ok, calls=calls, vcalls=vcalls, loops=loops, gaps=0, events=evs)
                else:
                    r = graph_search(root, executor, EVAL_BUDGET, rng, scorer)
            return task, s, r

        for task, s, r in pmap(run_ep, [(t, s) for t in TEST for s in range(EVAL_SEEDS)]):
            if True:
                agg["episodes"] += 1
                agg["success"] += r["success"]
                agg["calls"] += r["calls"]
                agg["vcalls"] += r["vcalls"]
                agg["verify_fail"] += r["vcalls"] - int(r["success"])
                agg["loops"] += r["loops"]
                agg["gaps"] += r["gaps"]
                if r["success"]:
                    agg["calls_success"].append(r["calls"])
                if s < 10:
                    episodes[f"{cname}|{task}|{s}"] = dict(controller=cname, task=task, seed=s, **r)
        n = agg["episodes"]
        res[cname] = dict(
            episodes=n, verified_success_rate=agg["success"] / n, success_ci95=wilson(agg["success"], n),
            mean_executor_calls_per_episode=agg["calls"] / n,
            calls_per_verified_success=(agg["calls"] / agg["success"]) if agg["success"] else None,
            median_calls_when_successful=float(np.median(agg["calls_success"])) if agg["calls_success"] else None,
            verify_calls=agg["vcalls"],
            verify_fail_rate=(agg["verify_fail"] / agg["vcalls"]) if agg["vcalls"] else None,
            loop_calls=agg["loops"], flag_gap_high_v=agg["gaps"], human_queries=0)
    return res, episodes


def to_sts(ep):
    """Export one episode in HF Session Trace Simple Format."""
    lines = [{"type": "session", "harness": "trace-critic", "id": f"{ep['controller']}-{ep['task']}-{ep['seed']}",
              "name": f"{ep['controller']} on {ep['task']} (seed {ep['seed']})"},
             {"type": "message", "message": {"role": "user", "content": TASKS[ep["task"]].goal}}]
    for i, e in enumerate(ep["events"]):
        if e.get("action") is None:
            lines.append({"type": "message", "message": {"role": "assistant", "content": e["decision"]}})
            continue
        tid = f"t{i}"
        lines.append({"type": "message", "message": {"role": "assistant", "content": e["decision"], "toolCalls": [
            {"id": tid, "function": {"name": "set_site", "arguments": json.dumps({"site": e["action"][0], "variant": e["action"][1]})}}]}})
        lines.append({"type": "message", "message": {"role": "tool", "toolCallId": tid, "content": json.dumps(
            {"visible_pass_frac": e.get("vis"), "critic_V": e.get("score"), "verified": e.get("verified")})}})
    return "\n".join(json.dumps(x) for x in lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--executor", choices=["sim", "nemotron"], default="sim")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    t0 = time.time()
    global OUT
    if args.executor == "sim":
        executor = SimulatedExecutor()
    else:
        executor = NemotronExecutor(args.model)
        globals().update(NEMOTRON_CFG)
        OUT = RUNS / "nemotron"
    RUNS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for t in TASKS:
        get_contract(t)
    print("verifier warm-up artifacts:", warm_cache(list(TASKS)))

    rows = {}
    for split, names in (("train", TRAIN), ("valid", VALID), ("test", TEST)):
        rows[split] = [r for n in names for r in collect(n, executor, split)]
        print(split, "labeled states:", len(rows[split]))
    with open(OUT / "labels.jsonl", "w") as f:
        for split in rows:
            for r in rows[split]:
                f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n")

    Xtr, ytr = XY(rows["train"])
    Xva, yva = XY(rows["valid"])
    Xte, _ = XY(rows["test"])
    # Feature-set selection on the VALID family only: full (struct + hashed source tokens) vs struct-only.
    from .search import N_STRUCT
    fsel = {}
    for fs, cols in (("full", slice(None)), ("struct", slice(0, N_STRUCT))):
        _, cv = C.train(Xtr[:, cols], ytr, Xva[:, cols], yva, opt="adamw")
        fsel[fs] = min(c[2] for c in cv)
    feat_set = min(fsel, key=fsel.get)
    cols = slice(None) if feat_set == "full" else slice(0, N_STRUCT)
    print("feature-set valid MSE:", fsel, "->", feat_set)
    Xtr, Xva, Xte = Xtr[:, cols], Xva[:, cols], Xte[:, cols]
    critics, crit_metrics, curves = {}, {}, {}
    for opt in ("adamw", "muon"):
        t1 = time.time()
        m, curve = C.train(Xtr, ytr, Xva, yva, opt=opt)
        m.cols = cols
        critics[opt] = m
        curves[opt] = curve
        crit_metrics[opt] = dict(train_seconds=round(time.time() - t1, 2),
                                 best_valid_mse=min(c[2] for c in curve),
                                 test=critic_metrics(C.predict(m, Xte), rows["test"]))
    # non-learned references on the same sealed test states
    vis = Xte[:, 0]
    crit_metrics["ref_visible_pass_frac"] = {"test": critic_metrics(vis, rows["test"])}
    crit_metrics["ref_train_mean"] = {"test": critic_metrics(np.full(len(Xte), ytr.mean()), rows["test"])}

    ctrl, episodes = evaluate(executor, critics)
    results = dict(
        note=("Executor is a seeded SIMULATED policy, not an LLM. Small-scale mechanism check, not a statistical "
              "claim." if args.executor == "sim" else
              f"Executor: {executor.policy_id} on Crusoe Serverless Inference. Small run (few episodes per task); "
              "a mechanism check, not a statistical claim."),
        executor=executor.policy_id, splits={"train": TRAIN, "valid": VALID, "test": TEST},
        config=dict(N_TRAJ=N_TRAJ, SRC_STEPS=SRC_STEPS, K_CONT=K_CONT, BUDGETS=BUDGETS,
                    EVAL_SEEDS=EVAL_SEEDS, EVAL_BUDGET=EVAL_BUDGET, branching_k=2, feat_dim=FEAT_DIM),
        labeled_states={k: len(v) for k, v in rows.items()},
        feature_selection_valid_mse=fsel, feature_set=feat_set,
        critic=crit_metrics, controllers=ctrl, wall_seconds=round(time.time() - t0, 1),
        executor_stats=dict(getattr(executor, "stats", {})))
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    (OUT / "loss_curves.json").write_text(json.dumps(curves))
    (OUT / "episodes.json").write_text(json.dumps(episodes))
    (OUT / "traces").mkdir(exist_ok=True)
    for k, ep in episodes.items():
        (OUT / "traces" / (k.replace("|", "__") + ".jsonl")).write_text(to_sts(ep))
    from .report import write_html
    write_html(results, curves, episodes, OUT)
    print(json.dumps({k: {kk: v[kk] for kk in ("verified_success_rate", "success_ci95", "calls_per_verified_success",
                                              "verify_fail_rate", "loop_calls")} for k, v in ctrl.items()}, indent=1))
    print("critic test mse:", {k: round(v["test"]["mse"], 4) for k, v in crit_metrics.items()})
    print("done in", results["wall_seconds"], "s")


if __name__ == "__main__":
    main()
