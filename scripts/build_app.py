"""Export real logged episodes for the TRACE app (site/public/app/data.json).

For every artifact that appears in an episode we store the rendered source and the result of
EVERY contract check (visible + hidden), computed by the same pytest verifier. Nothing is invented.

Usage: uv run python scripts/build_app.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trace_critic.env import verify  # noqa: E402
from trace_critic.tasks import TASKS  # noqa: E402
from trace_critic.tasks_v2 import FTASKS  # noqa: E402

OUT = ROOT / "site/public/app"


def main():
    episodes = json.loads((ROOT / "runs/episodes.json").read_text())
    results = json.loads((ROOT / "runs/results.json").read_text())
    artifacts, tasks = {}, {}
    for ep in episodes.values():
        t = TASKS[ep["task"]]
        tasks[t.name] = {"goal": t.goal, "criteria": [{"id": c, "visible": c in t.visible} for c in t.criteria],
                         "sites": len(t.sites)}
        for e in ep["events"]:
            if e.get("variant") is None:
                continue
            key = t.name + "|" + ",".join(map(str, e["variant"]))
            if key not in artifacts:
                h, checks = verify(t.name, tuple(e["variant"]))
                artifacts[key] = {"src": t.render(tuple(e["variant"])), "checks": checks, "hash": h}
    data = {
        "note": results["note"], "executor": results["executor"], "budget": results["config"]["EVAL_BUDGET"],
        "controllers": {k: {kk: v[kk] for kk in ("verified_success_rate", "success_ci95", "episodes",
                                                 "calls_per_verified_success", "verify_fail_rate")}
                        for k, v in results["controllers"].items()},
        "tasks": tasks, "artifacts": artifacts, "episodes": episodes,
    }
    # Real Pi Agent sessions (HF agent-traces), heuristic audit only.
    pi = json.loads((ROOT / "runs/pi_audit.json").read_text())
    for sess in pi["sessions"]:
        for e in sess["events"]:
            for k in ("args", "result", "detail"):
                if k in e and e[k]:
                    e[k] = e[k][:220]
    data["pi"] = pi
    # SYNTH: all traces with a flagged step + an equal number of clean ones (deterministic pick).
    syn = json.loads((ROOT / "runs/synth_audit.json").read_text())
    rows = {json.loads(l)["synth_id"]: json.loads(l) for l in open(ROOT / "data/synth/math_exercise_synth_001.jsonl")}
    bad = [t for t in syn["traces"] if t["n_bad"]]
    clean = [t for t in syn["traces"] if t["steps"] and not t["n_bad"]][: len(bad)]
    data["synth"] = {"summary": syn["summary"], "audit": json.loads((ROOT / "runs/synth_manual_audit.json").read_text()),
                     "traces": [{"synth_id": t["synth_id"], "query": rows[t["synth_id"]]["query"],
                                 "reasoning": rows[t["synth_id"]]["synthetic_reasoning"],
                                 "steps": t["steps"], "n_bad": t["n_bad"]} for t in bad + clean]}
    # Nemotron + DoD-from-prompt + TraceLoss (v2)
    v2 = ROOT / "runs/v2"
    eps = json.loads((v2 / "eval_episodes.json").read_text())
    for e in eps.values():
        for x in e["events"]:
            x.pop("src", None)
            if x.get("diff"):
                x["diff"] = x["diff"][:600]
    dod = json.loads((v2 / "dod.json").read_text())
    data["v2"] = {"eval": json.loads((v2 / "eval_results.json").read_text()), "episodes": eps,
                  "traceloss": json.loads((v2 / "traceloss_results.json").read_text()),
                  "pi_curves": json.loads((v2 / "pi_curves.json").read_text())["by_signal"],
                  "dod": {k: {kk: d[kk] for kk in ("code", "n_tests", "fail_on_buggy", "pass_on_reference",
                                                    "dropped_by_auditor", "attempt")} for k, d in dod.items()},
                  "goals": {k: FTASKS[k].goal for k in dod}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps(data, separators=(",", ":")))
    print("episodes:", len(episodes), "artifacts:", len(artifacts),
          "size KB:", (OUT / "data.json").stat().st_size // 1024)


if __name__ == "__main__":
    main()
