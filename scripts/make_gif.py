"""Render one real search episode from runs/episodes.json as an animated GIF (assets/trace.gif).

Usage: uv run python scripts/make_gif.py [--key "graph_learned_muon|median|1"]
Nothing is invented: every node, score, decision and budget value comes from the episode log.
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H = 960, 540
TREE = (40, 120, 640, 500)  # x0, y0, x1, y1
BG, INK, MUTED = "#f4f9ff", "#1f2937", "#6b7280"
DEC_COL = {"ROOT": "#6b7280", "CONTINUE": "#3b82f6", "BACKTRACK": "#f59e0b", "REPLAN(loop)": "#a855f7",
           "FLAG_GAP": "#dc2626", "VERIFIED": "#16a34a", "STOP_UNRESOLVED": "#111827"}
DEC_TXT = {"ROOT": "start from buggy checkpoint", "CONTINUE": "expand best branch",
           "BACKTRACK": "critic prefers an older checkpoint",
           "REPLAN(loop)": "exact repeat, no new evidence: drop it",
           "FLAG_GAP": "visible checks pass, verifier says NO",
           "VERIFIED": "independent pytest verifier: all checks pass",
           "STOP_UNRESOLVED": "budget exhausted"}


def font(size, bold=False):
    for p in (["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"]
              + ["/usr/share/fonts/truetype/dejavu/DejaVuSans" + ("-Bold" if bold else "") + ".ttf"]):
        try:
            return ImageFont.truetype(p, size, index=1 if bold and p.endswith(".ttc") else 0)
        except OSError:
            continue
    return ImageFont.load_default(size)


def v_color(v):
    """Critic value -> red (0) .. amber .. green (1)."""
    v = max(0.0, min(1.0, v))
    a, b = ((220, 38, 38), (245, 158, 11)) if v < 0.5 else ((245, 158, 11), (22, 163, 74))
    t = v * 2 if v < 0.5 else (v - 0.5) * 2
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def layout(events):
    depth, order = {}, {}
    for e in events:
        if e["id"] is None:
            continue
        depth[e["id"]] = 0 if e["parent"] is None else depth[e["parent"]] + 1
    per = {}
    for e in events:
        if e["id"] is None:
            continue
        d = depth[e["id"]]
        order[e["id"]] = per.get(d, 0)
        per[d] = per.get(d, 0) + 1
    md = max(depth.values()) or 1
    x0, y0, x1, y1 = TREE
    pos = {}
    for i, d in depth.items():
        pos[i] = (x0 + 20 + d / md * (x1 - x0 - 40), y0 + (order[i] + 0.5) / per[d] * (y1 - y0))
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="graph_learned_muon|median|1")
    args = ap.parse_args()
    ep = json.loads((ROOT / "runs/episodes.json").read_text())[args.key]
    results = json.loads((ROOT / "runs/results.json").read_text())
    budget = results["config"]["EVAL_BUDGET"]
    events = ep["events"]
    pos = layout(events)
    logo = Image.open(ROOT / "assets/logo_1024.png").convert("RGBA").resize((72, 72))
    f_title, f_big, f_txt, f_small = font(26, True), font(20, True), font(15), font(12)

    frames, durs = [], []
    for step in range(len(events)):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        im.paste(logo, (W - 92, 18), logo)
        d.text((40, 22), "Nesting Ralphs · TRACE", font=f_title, fill=INK)
        d.text((40, 58), f"Real episode replay: {ep['controller']} on '{ep['task']}' "
                         f"(seed {ep['seed']}) · simulated executor", font=f_small, fill=MUTED)
        d.text((40, 76), "Node colour = critic V(state, budget) ≈ P(verified success). "
                         "The critic steers; only pytest certifies.", font=f_small, fill=MUTED)

        shown = events[: step + 1]
        cur = events[step]
        # edges
        for e in shown:
            if e["id"] is None or e["parent"] is None:
                continue
            col = DEC_COL.get(e["decision"], MUTED)
            w = 3 if e is cur else 2
            d.line([pos[e["parent"]], pos[e["id"]]], fill=col, width=w)
        # nodes
        for e in shown:
            if e["id"] is None:
                continue
            x, y = pos[e["id"]]
            if e["decision"] == "REPLAN(loop)":
                d.line([(x - 5, y - 5), (x + 5, y + 5)], fill=DEC_COL["REPLAN(loop)"], width=3)
                d.line([(x - 5, y + 5), (x + 5, y - 5)], fill=DEC_COL["REPLAN(loop)"], width=3)
                continue
            r = 11
            fill = v_color(e["score"]) if isinstance(e.get("score"), (int, float)) else (200, 200, 200)
            d.ellipse([x - r, y - r, x + r, y + r], fill=fill, outline="white", width=2)
            if e["decision"] == "FLAG_GAP":
                d.ellipse([x - r - 4, y - r - 4, x + r + 4, y + r + 4], outline=DEC_COL["FLAG_GAP"], width=3)
            if e["decision"] == "VERIFIED":
                d.ellipse([x - r - 6, y - r - 6, x + r + 6, y + r + 6], outline=DEC_COL["VERIFIED"], width=4)
            if e is cur:
                d.ellipse([x - r - 9, y - r - 9, x + r + 9, y + r + 9], outline=INK, width=1)
        # side panel
        px = 680
        d.rounded_rectangle([px - 10, 110, W - 20, 505], radius=12, fill="white", outline="#dbe4ee")
        d.text((px + 6, 124), f"step {step + 1}/{len(events)}", font=f_small, fill=MUTED)
        col = DEC_COL.get(cur["decision"], INK)
        d.text((px + 6, 144), cur["decision"], font=f_big, fill=col)
        d.text((px + 6, 172), DEC_TXT.get(cur["decision"], ""), font=f_small, fill=INK)
        y = 204
        if cur.get("action"):
            d.text((px + 6, y), f"action: edit site {cur['action'][0]} -> variant {cur['action'][1]}",
                   font=f_txt, fill=INK)
            y += 24
        if isinstance(cur.get("score"), (int, float)):
            d.text((px + 6, y), f"critic V = {cur['score']:.2f}", font=f_txt, fill=INK)
            y += 24
        if cur.get("vis") is not None:
            d.text((px + 6, y), f"visible checks passing: {cur['vis']:.0%}", font=f_txt, fill=INK)
            y += 24
        if cur.get("verified") is not None:
            d.text((px + 6, y), f"verifier (all checks): {'PASS' if cur['verified'] else 'FAIL'}",
                   font=f_txt, fill=DEC_COL["VERIFIED"] if cur["verified"] else DEC_COL["FLAG_GAP"])
        used = max((e.get("calls") or 0) for e in shown)
        d.text((px + 6, 400), f"executor budget: {used}/{budget} calls", font=f_txt, fill=INK)
        d.rounded_rectangle([px + 6, 424, W - 36, 440], radius=8, fill="#e5e7eb")
        d.rounded_rectangle([px + 6, 424, px + 6 + (W - 42 - px) * used / budget, 440], radius=8, fill="#3b82f6")
        loops = sum(e["decision"] == "REPLAN(loop)" for e in shown)
        d.text((px + 6, 452), f"loops caught: {loops}", font=f_small, fill=DEC_COL["REPLAN(loop)"])
        d.text((px + 6, 470), "every executed branch is charged", font=f_small, fill=MUTED)
        # legend
        lx = 40
        for name in ("CONTINUE", "BACKTRACK", "REPLAN(loop)", "FLAG_GAP", "VERIFIED"):
            d.line([(lx, 520), (lx + 16, 520)], fill=DEC_COL[name], width=3)
            d.text((lx + 20, 512), name, font=f_small, fill=INK)
            lx += 20 + d.textlength(name, font=f_small) + 16
        d.text((lx + 4, 512), "node: V", font=f_small, fill=INK)
        for i in range(5):
            d.ellipse([lx + 56 + i * 14, 515, lx + 66 + i * 14, 525], fill=v_color(i / 4))
        d.text((lx + 130, 512), "0 .. 1", font=f_small, fill=MUTED)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
        durs.append(2600 if step == len(events) - 1 else (1100 if cur["decision"] in DEC_TXT and cur["decision"] not in ("CONTINUE", "REPLAN(loop)") else 650))

    out = ROOT / "assets/trace.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=durs, loop=0, optimize=True)
    print(out, len(frames), "frames", out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
