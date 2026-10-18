"""Apply the trained TraceLoss to real Pi Agent sessions (HF agent-traces). OUT OF DISTRIBUTION demo:
DoD = first user message, steps = tool call + result. No verifier outcome exists for these traces.
Usage: PYTHONPATH=. uv run python -m trace_critic.pi_curves
"""
import json
from pathlib import Path
import numpy as np
import torch
from .traceloss import MAX_STEPS, TraceLoss, batchify, embed

OUT = Path("runs/v2")


def main():
    pi = json.loads(Path("runs/pi_audit.json").read_text())
    m = TraceLoss()
    m.load_state_dict(torch.load(OUT / "traceloss.pt"))
    m.eval()
    sessions = []
    for s in pi["sessions"]:
        lines = [json.loads(l) for l in open(Path("data/pi") / s["file"])]
        first = next((x for x in lines if x.get("type") == "message" and x["message"]["role"] == "user"), None)
        goal = ""
        if first:
            c = first["message"]["content"]
            goal = c if isinstance(c, str) else " ".join(p.get("text", "") for p in c if isinstance(p, dict))
        evs = [e for e in s["events"] if e["decision"] != "USER"]
        if not evs:
            continue
        texts = [f"ACTION: {e.get('tool') or ''} {e.get('args', '') or e.get('detail', '')}\nRESULT: "
                 f"{'ERROR ' if e.get('error') else ''}{e.get('result', '')}\nSIGNAL: {e['decision']}" for e in evs]
        Z = embed(texts)
        zd = embed([f"TASK: {goal[:1500]}\nDEFINITION OF DONE: the request is completed and checks pass"])[0]
        items = [{"zdod": zd, "zs": Z[max(0, t + 1 - MAX_STEPS):t + 1], "B": 4} for t in range(len(evs))]
        with torch.no_grad():
            V = np.concatenate([m(*batchify(items[k:k + 64])).numpy() for k in range(0, len(items), 64)])
        sessions.append({"file": s["file"], "model": s["model"], "goal": goal[:300],
                         "V": [round(float(v), 4) for v in V], "decisions": [e["decision"] for e in evs]})
    # simple association check: mean V right after failing vs passing checks / errors
    agg = {}
    for s in sessions:
        for v, d in zip(s["V"], s["decisions"]):
            agg.setdefault(d, []).append(v)
    summary = {d: {"n": len(v), "mean_V": round(float(np.mean(v)), 4)} for d, v in agg.items()}
    json.dump({"note": "Out-of-distribution demo; no verifier labels exist for these traces.", "by_signal": summary,
               "sessions": sessions}, open(OUT / "pi_curves.json", "w"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
