"""TRACE: static HTML replay of the experiment. Every number comes from runs/*.json."""

import html
import json

from .env import RUNS

COLORS = {"ROOT": "#888", "CONTINUE": "#3b82f6", "BACKTRACK": "#f59e0b", "VERIFIED": "#16a34a",
          "FLAG_GAP": "#dc2626", "REPLAN(loop)": "#a855f7", "STOP_UNRESOLVED": "#111"}


def _curve_svg(curves, w=520, h=180):
    pts = [v for c in curves.values() for _, a, b in c for v in (a, b)]
    hi = max(pts) or 1
    n = max(len(c) for c in curves.values())
    out = [f'<svg width="{w}" height="{h}" style="background:#fafafa;border:1px solid #ddd">']
    for (name, c), col in zip(curves.items(), ["#3b82f6", "#dc2626"]):
        for idx, dash in ((1, ""), (2, "4,3")):
            p = " ".join(f"{10 + e / max(1, n - 1) * (w - 20):.1f},{h - 10 - r[idx] / hi * (h - 20):.1f}"
                         for e, r in enumerate(c))
            out.append(f'<polyline fill="none" stroke="{col}" stroke-dasharray="{dash}" points="{p}"/>')
    out.append(f'<text x="12" y="14" font-size="11">MSE (solid=train, dashed=valid) · blue=AdamW · red=Muon+AdamW · max {hi:.3f}</text></svg>')
    return "".join(out)


def _episode_svg(ep, w=520, h=150):
    ev = [e for e in ep["events"] if e.get("calls") is not None or ep["controller"] == "ralph"]
    out = [f'<svg width="{w}" height="{h}" style="background:#fff;border:1px solid #ddd">',
           f'<line x1="10" x2="{w - 10}" y1="{h - 20}" y2="{h - 20}" stroke="#ccc"/>']
    n = max(1, len(ev))
    prev = None
    for i, e in enumerate(ev):
        x = 15 + i / n * (w - 30)
        y_vis = h - 20 - (e["vis"] or 0) * (h - 40) if e.get("vis") is not None else None
        y_v = h - 20 - e["score"] * (h - 40) if isinstance(e.get("score"), float) and 0 <= e["score"] <= 1 else None
        col = COLORS.get(e["decision"], "#666")
        if y_v is not None:
            out.append(f'<circle cx="{x:.1f}" cy="{y_v:.1f}" r="4" fill="{col}"><title>{html.escape(e["decision"])} V={e["score"]:.2f}</title></circle>')
            if prev:
                out.append(f'<line x1="{prev[0]:.1f}" y1="{prev[1]:.1f}" x2="{x:.1f}" y2="{y_v:.1f}" stroke="#bbb"/>')
            prev = (x, y_v)
        if y_vis is not None:
            out.append(f'<rect x="{x - 3:.1f}" y="{y_vis - 3:.1f}" width="6" height="6" fill="none" stroke="{col}"><title>visible pass {e["vis"]:.2f}</title></rect>')
        if e["decision"] in ("VERIFIED", "FLAG_GAP", "BACKTRACK", "REPLAN(loop)", "STOP_UNRESOLVED"):
            out.append(f'<text x="{x:.1f}" y="{12 + (i % 3) * 11}" font-size="9" fill="{col}">{html.escape(e["decision"])}</text>')
    out.append(f'<text x="12" y="{h - 5}" font-size="10">dots = critic/rule score · squares = visible-check pass fraction · x = step</text></svg>')
    return "".join(out)


def write_html(results, curves, episodes, out=RUNS):
    ctrl = results["controllers"]
    rows = "".join(
        f"<tr><td>{k}</td><td>{v['verified_success_rate']:.3f}</td><td>{v['success_ci95']}</td>"
        f"<td>{v['calls_per_verified_success'] and round(v['calls_per_verified_success'], 2)}</td>"
        f"<td>{v['verify_fail_rate'] if v['verify_fail_rate'] is None else round(v['verify_fail_rate'], 3)}</td>"
        f"<td>{v['loop_calls']}</td><td>{v['episodes']}</td></tr>" for k, v in ctrl.items())
    crit = "".join(
        f"<tr><td>{k}</td><td>{v['test']['mse']:.4f}</td><td>{v['test']['pairwise_rank_acc']}</td>"
        f"<td>{v['test']['mse_all_visible_pass']}</td><td>{v.get('train_seconds', '')}</td></tr>"
        for k, v in results["critic"].items())
    eps = "".join(
        f"<details><summary>{html.escape(k)} — {'VERIFIED' if ep['success'] else 'UNRESOLVED'} · {ep['calls']} calls</summary>"
        f"{_episode_svg(ep)}<pre>{html.escape(json.dumps(ep['events'], indent=0)[:3000])}</pre></details>"
        for k, ep in episodes.items())
    doc = f"""<!doctype html><meta charset="utf-8"><title>Nesting Ralphs · TRACE</title><link rel="icon" href="ASSETS/favicon.png">
<style>body{{font:14px/1.45 system-ui;max-width:980px;margin:24px auto;padding:0 16px}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:4px 8px;font-size:13px}}
.warn{{background:#fff7e6;border:1px solid #f5c26b;padding:8px}}pre{{font-size:11px;max-height:240px;overflow:auto}}</style>
<img src="ASSETS/banner.jpg" alt="" style="width:100%;border-radius:8px">
<h1><img src="ASSETS/logo.png" alt="" width="56" style="vertical-align:middle"> Nesting Ralphs · TRACE</h1>
<p class="warn"><b>{html.escape(results['note'])}</b> Executor: <code>{html.escape(results['executor'])}</code>.
All numbers below are read from <code>runs/results.json</code>.</p>
<p>The critic steers; it never certifies. Only the independent pytest verifier sets VERIFIED.</p>
<h2>Controllers (sealed test tasks: {', '.join(results['splits']['test'])}; budget {results['config']['EVAL_BUDGET']} executor calls)</h2>
<table><tr><th>controller</th><th>verified success</th><th>95% CI (Wilson)</th><th>calls / verified success</th>
<th>verify-fail rate</th><th>loop calls</th><th>episodes</th></tr>{rows}</table>
<h2>Critic on sealed test states</h2>
<p>Labeled states: {results['labeled_states']}</p>
<table><tr><th>model</th><th>MSE</th><th>pairwise rank acc</th><th>MSE (all visible pass)</th><th>train s</th></tr>{crit}</table>
<h2>Training curves</h2>{_curve_svg(curves)}
<h2>Episode replays</h2>{eps}"""
    rel = "../assets/" if out == RUNS else "../../assets/"
    (out / "trace.html").write_text(doc.replace("ASSETS/", rel))
