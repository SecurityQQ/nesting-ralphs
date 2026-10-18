"""Controller comparison with the real Nemotron executor on the sealed test tasks, equal budget.

ralph            : bounded Ralph loop
graph_rules      : best-first graph search, score = DoD pass fraction - 0.2 * verifier rejections
graph_traceloss  : same graph search, score = TraceLoss V(DoD, trace embeddings, budget left)
Usage: PYTHONPATH=. uv run python -m trace_critic.eval_v2
"""

import asyncio
import json
import math
import time
from pathlib import Path

import torch

from . import nemo
from .nemo import initial_state, maybe_verify, ralph, step, step_text, USAGE
from .tasks_v2 import FTASKS, TEST2
from .traceloss import MAX_STEPS, TraceLoss, batchify, dod_text, embed

OUT = Path("runs/v2")
BUDGET, SEEDS, K = 6, 8, 2
_cache = {}


def emb(texts):
    miss = [t for t in texts if t not in _cache]
    if miss:
        for t, z in zip(miss, embed(miss)):
            _cache[t] = z
    return [_cache[t] for t in texts]


def make_traceloss_scorer(D):
    m = TraceLoss()
    m.load_state_dict(torch.load(OUT / "traceloss.pt"))
    m.eval()

    def score(task, state, budget_left):
        steps = state["steps"][-MAX_STEPS:]
        zd = emb([dod_text(task, D[task.name])])[0]
        zs = emb([step_text(s) for s in steps])
        item = {"zdod": zd, "zs": __import__("numpy").stack(zs), "B": budget_left}
        with torch.no_grad():
            return float(m(*batchify([item]))[0])
    return score


def rule_score(task, state, budget_left):
    d = state["steps"][-1]["dod"]
    return d["passed"] / max(1, d["total"]) - 0.2 * state["verify_fails"]


def ev(state, parent_id, node_id, decision, score, calls):
    s = state["steps"][-1]
    return {"id": node_id, "parent": parent_id, "decision": decision, "score": score, "calls": calls,
            "thought": s["thought"], "diff": s["diff"][:900], "dod": s["dod"], "verified": s.get("verified"),
            "src": s.get("src_snapshot", "")[:4000]}


async def graph_search(task, dod, scorer, budget=BUDGET):
    root = initial_state(task, dod)
    ids = iter(range(10 ** 6))
    calls = vcalls = loops = 0
    rid = next(ids)
    root["_id"], root["_exp"] = rid, 0
    root["_score"] = scorer(task, root, budget)
    events = [ev(root, None, rid, "ROOT", root["_score"], 0)]
    ok, did = await maybe_verify(task, root)
    vcalls += did
    seen = {root["steps"][-1]["artifact"]}
    frontier, current = [root], root
    while calls + K <= budget and frontier:
        best = max(frontier, key=lambda n: n["_score"] - 0.05 * n["_exp"])
        decision = "CONTINUE" if best is current or best["steps"][:-1] == current["steps"] else "BACKTRACK"
        best["_exp"] += 1
        current = best
        children = await asyncio.gather(*[step(task, dod, {"src": best["src"], "verify_fails": best["verify_fails"],
                                                           "steps": best["steps"]}) for _ in range(K)])
        calls += K
        for ch in children:
            cid = next(ids)
            art = ch["steps"][-1]["artifact"]
            if art in seen:
                loops += 1
                events.append(ev(ch, best["_id"], cid, "REPLAN(loop)", None, calls))
                continue
            seen.add(art)
            ok, did = await maybe_verify(task, ch)
            vcalls += did
            ch["_id"], ch["_exp"] = cid, 0
            ch["_score"] = scorer(task, ch, budget - calls)
            if ok:
                events.append(ev(ch, best["_id"], cid, "VERIFIED", ch["_score"], calls))
                return dict(success=True, calls=calls, vcalls=vcalls, loops=loops, events=events)
            events.append(ev(ch, best["_id"], cid, "FLAG_GAP" if did else decision, ch["_score"], calls))
            frontier.append(ch)
        frontier.sort(key=lambda n: n["_score"], reverse=True)
        del frontier[8:]
    events.append({"id": None, "parent": None, "decision": "STOP_UNRESOLVED", "score": None, "calls": calls})
    return dict(success=False, calls=calls, vcalls=vcalls, loops=loops, events=events)


async def run_ralph(task, dod, budget=BUDGET):
    ok, calls, vcalls, st = await ralph(task, dod, initial_state(task, dod), budget)
    events, seen, loops = [], set(), 0
    for i, s in enumerate(st["steps"]):
        dec = "ROOT" if i == 0 else "VERIFIED" if s.get("verified") else "FLAG_GAP" if s.get("verified") is False else "CONTINUE"
        if i and s["artifact"] in seen and dec == "CONTINUE":
            loops += 1
            dec = "REPEAT"
        seen.add(s["artifact"])
        events.append(ev({"steps": [s]}, i - 1 if i else None, i, dec, None, i))
    if not ok:
        events.append({"id": None, "parent": None, "decision": "STOP_UNRESOLVED", "score": None, "calls": calls})
    return dict(success=ok, calls=calls, vcalls=vcalls, loops=loops, events=events)


def wilson(k, n, z=1.96):
    if not n:
        return [0, 0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


async def main():
    t0 = time.time()
    D = json.loads((OUT / "dod.json").read_text())
    tl = make_traceloss_scorer(D)
    jobs = [(c, n, s) for c in ("ralph", "graph_rules", "graph_traceloss") for n in TEST2 for s in range(SEEDS)]

    async def run(c, n, s):
        task = FTASKS[n]
        if c == "ralph":
            return await run_ralph(task, D[n])
        return await graph_search(task, D[n], rule_score if c == "graph_rules" else tl)

    res = await asyncio.gather(*[run(*j) for j in jobs])
    summary = {}
    for c in ("ralph", "graph_rules", "graph_traceloss"):
        rs = [(j, r) for j, r in zip(jobs, res) if j[0] == c]
        n = len(rs)
        k = sum(r["success"] for _, r in rs)
        vc = sum(r["vcalls"] for _, r in rs)
        summary[c] = {"episodes": n, "verified_success_rate": k / n, "success_ci95": wilson(k, n),
                      "mean_executor_calls": sum(r["calls"] for _, r in rs) / n,
                      "calls_per_verified_success": sum(r["calls"] for _, r in rs) / k if k else None,
                      "verifier_calls": vc, "false_done": vc - k,
                      "false_done_rate": (vc - k) / vc if vc else None,
                      "loops": sum(r["loops"] for _, r in rs),
                      "per_task": {t: sum(r["success"] for j, r in rs if j[1] == t) for t in TEST2}}
    episodes = {f"{c}|{n}|{s}": {"controller": c, "task": n, "seed": s, **r} for (c, n, s), r in zip(jobs, res)}
    out = {"executor": nemo.EXEC_MODEL, "dod_model": nemo.DOD_MODEL, "budget": BUDGET, "seeds": SEEDS,
           "branching_k": K, "test_tasks": TEST2, "controllers": summary, "usage": USAGE,
           "seconds": round(time.time() - t0)}
    (OUT / "eval_results.json").write_text(json.dumps(out, indent=1))
    (OUT / "eval_episodes.json").write_text(json.dumps(episodes))
    for c, v in summary.items():
        print(c, {k: v[k] for k in ("verified_success_rate", "success_ci95", "calls_per_verified_success",
                                    "false_done", "loops", "per_task")})
    print(USAGE, out["seconds"], "s")


if __name__ == "__main__":
    asyncio.run(main())
