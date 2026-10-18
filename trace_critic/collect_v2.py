"""Collect real Nemotron trajectories and rollout labels.

For each task: N_SRC source trajectories (bounded Ralph, budget SRC_B). Every non-verified prefix state
gets K continuations with budget B; label q = verified successes / K (hidden tests decide success).
Usage: PYTHONPATH=. uv run python -m trace_critic.collect_v2
"""
import asyncio, json, time, zlib
from pathlib import Path
from . import nemo
from .nemo import initial_state, ralph, USAGE
from .tasks_v2 import FTASKS, TRAIN2, VALID2, TEST2

N_SRC, SRC_B, K, BUDGETS = 6, 6, 3, (1, 2, 4)
OUT = Path("runs/v2")


def split_of(n):
    return "train" if n in TRAIN2 else "valid" if n in VALID2 else "test"


async def main():
    t0 = time.time()
    D = json.loads((OUT / "dod.json").read_text())
    jobs = [(n, s) for n in FTASKS for s in range(N_SRC)]
    src = await asyncio.gather(*[ralph(FTASKS[n], D[n], initial_state(FTASKS[n], D[n]), SRC_B) for n, s in jobs])
    print("source trajectories", len(src), "success", sum(r[0] for r in src), round(time.time() - t0), "s", USAGE)
    states = []
    for (n, s), (ok, calls, v, final) in zip(jobs, src):
        for t in range(len(final["steps"])):
            if t == 0 and s > 0:
                continue  # root is identical across trajectories of a task: keep one
            st = final["steps"][t]
            if st.get("verified"):
                break  # terminal
            prefix = {"src": None, "verify_fails": sum(1 for x in final["steps"][:t + 1] if x.get("verified") is False),
                      "steps": final["steps"][:t + 1]}
            states.append({"task": n, "split": split_of(n), "traj": s, "t": t, "prefix": prefix,
                           "B": BUDGETS[zlib.crc32(f"{n}|{s}|{t}".encode()) % len(BUDGETS)],
                           "src_final_ok": ok})
    # recover src for each prefix by replaying the diffs is fragile: store src snapshots instead
    snaps = {}
    for (n, s), (ok, calls, v, final) in zip(jobs, src):
        snaps[(n, s)] = final
    print("labeled states to collect", len(states))

    async def label(st):
        task, dod = FTASKS[st["task"]], D[st["task"]]
        state = {"src": st["src"], "verify_fails": st["prefix"]["verify_fails"], "steps": st["prefix"]["steps"]}
        rs = await asyncio.gather(*[ralph(task, dod, json.loads(json.dumps(state)), st["B"]) for _ in range(K)])
        return [int(r[0]) for r in rs], [r[1] for r in rs]

    for st in states:
        st["src"] = st["prefix"]["steps"][-1]["src_snapshot"]
    res = await asyncio.gather(*[label(st) for st in states])
    with open(OUT / "states.jsonl", "w") as f:
        for st, (ys, cs) in zip(states, res):
            st["successes"], st["k"], st["q"], st["cont_calls"] = sum(ys), K, sum(ys) / K, cs
            f.write(json.dumps(st) + "\n")
    nemo.save_json(OUT / "collect_meta.json", {"N_SRC": N_SRC, "SRC_B": SRC_B, "K": K, "BUDGETS": BUDGETS,
                   "exec_model": nemo.EXEC_MODEL, "usage": USAGE, "seconds": round(time.time() - t0),
                   "source": [{"task": n, "traj": s, "success": r[0], "calls": r[1], "steps": r[3]["steps"]}
                              for (n, s), r in zip(jobs, src)]})
    print("done", len(states), "states", round(time.time() - t0), "s", USAGE)

asyncio.run(main())
