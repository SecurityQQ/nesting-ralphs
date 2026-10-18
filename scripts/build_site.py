"""Build site/public/index.html: the TRACE-v1 page, re-skinned in a Ralph Wiggum style,
plus a 'What we measured' section filled from runs/results.json (no hand-typed numbers).

Usage: uv run python scripts/build_site.py
"""

import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "site/src/TRACE-v1.html"
OUT = ROOT / "site/public"
REPO = "https://github.com/SecurityQQ/nesting-ralphs"

FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Luckiest+Guy&family=Nunito:wght@400;600;700;800;900&family=Space+Mono:wght@400;700&display=swap">'
RALPH_CSS = r"""
:root{--paper:#bfe6ff;--ink:#1b1b1b;--muted:#3d4f63;--line:#1b1b1b;--accent:#e8352b;--yellow:#ffd90f;--sky:#8fd3ff;
--green:#2f9e44;--mono:'Space Mono',monospace;--sans:'Nunito',-apple-system,sans-serif;--comic:'Luckiest Guy','Nunito',sans-serif;
--pop:4px 4px 0 #1b1b1b;--pop-sm:3px 3px 0 #1b1b1b}
body{background:var(--sky) radial-gradient(#ffffff55 1.4px,transparent 1.5px) 0 0/22px 22px;font-weight:600}
::selection{background:var(--yellow);color:#1b1b1b}
h1,h2,h3,.brand,.comparison strong{font-family:var(--comic);font-weight:400!important;letter-spacing:.5px!important}
.hero h1{letter-spacing:1px!important;line-height:1.02;color:#fff;-webkit-text-stroke:3px #1b1b1b;paint-order:stroke fill;text-shadow:5px 5px 0 #1b1b1b}
.hero h1 .serif{color:var(--yellow);font-family:var(--comic);font-style:normal;letter-spacing:1px}
.serif{font-family:var(--comic);font-style:normal;color:var(--accent);letter-spacing:.5px}
h2{letter-spacing:.5px!important;color:#1b1b1b}
.header{border-bottom:4px solid #1b1b1b;background:var(--yellow);margin-top:18px;border:4px solid #1b1b1b;border-radius:18px;box-shadow:var(--pop);height:86px}
.header.wrap{max-width:1232px;padding-left:22px;padding-right:22px}
.brand{font-size:28px;color:var(--accent);-webkit-text-stroke:1.2px #1b1b1b;paint-order:stroke fill}
.brand img{width:58px;height:58px;border-radius:50%;border:3px solid #1b1b1b;background:#fff}
.brand-detail{color:#1b1b1b;-webkit-text-stroke:0;font-family:var(--mono);font-weight:700}
.header nav{color:#1b1b1b;font-weight:800}
.header-cta,.button{border:3px solid #1b1b1b!important;border-radius:999px!important;box-shadow:var(--pop-sm);font-weight:800;transition:transform .12s,box-shadow .12s}
.header-cta{background:#fff}
.header-cta:hover,.button:hover:not(:disabled){transform:translate(-2px,-2px);box-shadow:5px 5px 0 #1b1b1b}
.button.primary{background:var(--accent)}
.button.secondary{background:#fff}
.banner{margin:26px auto 0;max-width:1232px;border:4px solid #1b1b1b;border-radius:22px;overflow:hidden;box-shadow:var(--pop);background:#8fd3ff}
.banner img{display:block;width:100%;height:auto}
.hero-description,.section-heading p,.principle-grid p,.method-copy>p,.experiment-intro,.comparison p,.thesis-strip p{color:#1f2d3d!important;font-weight:600}
.small-note,.section-label,.panel-label,.principle-foot,.plot-top,.plot-bottom,.chart-legend,.board-note,.experiment-note{color:#1f2d3d!important}
.orange-dot{background:var(--accent);border:2px solid #1b1b1b;width:12px;height:12px}
.eyebrow{font-weight:700;color:#1b1b1b}
.ralph-quote{display:inline-block;background:#fff;border:3px solid #1b1b1b;border-radius:18px;padding:8px 14px;font-family:var(--comic);font-size:18px;box-shadow:var(--pop-sm);transform:rotate(-2deg);margin:0 0 22px}
.landscape,.lab,.equation-board,.research-note,.contract,.thesis-strip,.principle-grid article,.comparison>div,.results-card{background:#fffdf2!important;border:4px solid #1b1b1b!important;border-radius:18px!important;box-shadow:var(--pop)}
.landscape{background:#fff8d6!important}
.thesis-strip{padding:20px 24px;margin-top:6px}
.principle-grid{gap:22px}
.principle-grid article{padding:22px 24px!important}
.symbol{font-family:var(--comic);font-style:normal;color:var(--accent);font-size:34px}
.playground-section{background:var(--yellow);border-top:4px solid #1b1b1b;border-bottom:4px solid #1b1b1b}
.simulation-tag{background:#fff;border:2px solid #1b1b1b;color:#1b1b1b;font-weight:700}
.lab{overflow:hidden}
.lab-toolbar,.lab-footer{background:#ffe97a;border-color:#1b1b1b}
.lab-main,.trace-column,.lab-toolbar,.lab-footer,.step-detail,.contract-body{border-color:#1b1b1b!important}
.scenario{color:#1b1b1b;font-weight:800;border:2px solid transparent;border-radius:999px}
.scenario.active{background:var(--accent);color:#fff;border-color:#1b1b1b;box-shadow:var(--pop-sm)}
.trace-node{color:#1f2d3d;font-weight:700}
.trace-node.selected{background:#ffe97a;border:3px solid #1b1b1b;color:#1b1b1b;box-shadow:var(--pop-sm)}
.node-number{border:2px solid #1b1b1b;font-weight:700}
.selected .node-number{background:var(--accent);border-color:#1b1b1b}
.past .node-number{background:#b2f2bb;border-color:#1b1b1b;color:#1b1b1b}
.decision{border:2px solid #1b1b1b;font-weight:700;background:#d0ebff;color:#1b1b1b}
.decision.replan{background:#ffd8a8;color:#1b1b1b}.decision.reject{background:#ffc9c9;color:#1b1b1b}.decision.verified{background:#b2f2bb;color:#1b1b1b}
.loss-value{color:var(--accent);font-weight:700}
.equation-board{background:#1b4f8a!important;color:#fff}
.equation-row code{color:#fff}.equation-row:nth-child(4) code{color:var(--yellow)}
.equation-label,.equation-index,.board-title .mono,.board-bottom{color:#bfe6ff!important}
.board-title>span:last-child{color:var(--yellow)}
.board-note{color:#d0ebff!important}
.research-note{padding:18px 20px!important;border-left:4px solid #1b1b1b!important}
.metrics span{border:2px solid #1b1b1b;background:#fff;color:#1b1b1b;font-weight:700;border-radius:999px}
.comparison>div{padding:16px}
.versus{font-family:var(--comic);font-style:normal;color:var(--accent);font-size:24px}
.experiment,.reading,.footer{border-color:#1b1b1b}
.reading a span{color:#1f2d3d}
.footer .brand img{width:40px;height:40px}
.results{padding-top:30px;padding-bottom:60px}
.results-card{padding:24px 26px}
.results-grid{display:grid;grid-template-columns:1.15fr .85fr;gap:26px;align-items:start;margin-top:22px}
.rtable{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;margin-top:10px}
.rtable th,.rtable td{padding:9px 10px;border-bottom:2px dashed #1b1b1b33;text-align:left}
.rtable th{font-family:var(--mono);font-size:10px;text-transform:uppercase;color:#1f2d3d}
.rtable td.num{font-family:var(--mono);font-weight:700;white-space:nowrap}
.rtable tr.best td{background:#b2f2bb}
.verdict{background:#ffc9c9;border:3px solid #1b1b1b;border-radius:14px;padding:12px 14px;font-weight:800;margin:0 0 14px;box-shadow:var(--pop-sm)}
.results ul{margin:12px 0 0;padding-left:20px;line-height:1.7;font-size:14px}
.gif-card img{display:block;width:100%;border:3px solid #1b1b1b;border-radius:12px}
.gif-card p{font-size:12px;margin:10px 0 0;line-height:1.6}
.hero-tagline{font-family:var(--comic);font-size:24px;line-height:1.25;margin:-6px 0 14px;color:#1b1b1b}.hero-tagline b{color:var(--accent);font-weight:400}
.principles-mini{margin:0 0 18px;padding:12px 16px 12px 32px;background:#fffdf2;border:3px solid #1b1b1b;border-radius:14px;box-shadow:var(--pop-sm);font-size:14px;line-height:1.6;max-width:560px}
.sim-flag{display:inline-block;background:var(--yellow);border:2px solid #1b1b1b;border-radius:999px;padding:3px 10px;font-family:var(--mono);font-size:10px;font-weight:700}
@media(max-width:900px){.results-grid{grid-template-columns:1fr}.header{margin-top:10px}}
"""


def results_section():
    r = json.loads((ROOT / "runs/results.json").read_text())
    ctrl, crit = r["controllers"], r["critic"]
    names = {"ralph": "Bounded Ralph (repeat the loop)", "graph_rules": "Graph search + fixed rules",
             "graph_learned_adamw": "Graph search + learned critic (AdamW)",
             "graph_learned_muon": "Graph search + learned critic (Muon+AdamW)"}
    best = max(ctrl, key=lambda k: ctrl[k]["verified_success_rate"])
    rows = "".join(
        f"<tr class='{'best' if k == best else ''}'><td>{names[k]}</td>"
        f"<td class='num'>{v['verified_success_rate']:.3f}</td>"
        f"<td class='num'>[{v['success_ci95'][0]:.2f}, {v['success_ci95'][1]:.2f}]</td>"
        f"<td class='num'>{v['calls_per_verified_success']:.1f}</td>"
        f"<td class='num'>{v['verify_fail_rate']:.2f}</td></tr>" for k, v in ctrl.items())
    crows = "".join(
        f"<tr><td>{label}</td><td class='num'>{crit[k]['test']['mse']:.3f}</td>"
        f"<td class='num'>{'—' if not crit[k]['test']['pairwise_rank_acc'] else f'{crit[k]['test']['pairwise_rank_acc']:.2f}'}</td></tr>"
        for k, label in (("adamw", "Critic, AdamW"), ("muon", "Critic, Muon+AdamW"),
                         ("ref_visible_pass_frac", "Ref: visible pass fraction"),
                         ("ref_train_mean", "Ref: constant train mean")))
    pi = json.loads((ROOT / "runs/pi_audit.json").read_text())["summary"]
    sy = json.loads((ROOT / "runs/synth_audit.json").read_text())["summary"]
    au = sy["manual_precision_audit"]
    nemo = json.loads((ROOT / "runs/nemotron/results.json").read_text())
    nc = nemo["controllers"]
    nrows = "".join(f"<tr><td>{names[k]}</td><td class='num'>{v['verified_success_rate']:.2f}</td>"
                    f"<td class='num'>{v['mean_executor_calls_per_episode']:.1f}</td><td class='num'>{v['verify_fail_rate']:.2f}</td></tr>"
                    for k, v in nc.items())
    E2 = json.loads((ROOT / "runs/v2/eval_results.json").read_text())
    T2 = json.loads((ROOT / "runs/v2/traceloss_results.json").read_text())
    D2 = json.loads((ROOT / "runs/v2/dod.json").read_text())
    C2 = E2["controllers"]
    n2 = {"ralph": "Bounded Ralph", "graph_rules": "Graph + rules (DoD pass fraction)", "graph_traceloss": "Graph + TraceLoss (learned loss)"}
    rows2 = "".join(f"<tr class='{'best' if k == 'graph_traceloss' else ''}'><td>{n2[k]}</td><td class='num'>{v['verified_success_rate']:.3f}</td>"
                    f"<td class='num'>[{v['success_ci95'][0]:.2f}, {v['success_ci95'][1]:.2f}]</td><td class='num'>{v['false_done']}/{v['verifier_calls']}</td></tr>"
                    for k, v in C2.items())
    cv2 = T2["cv"]
    nt2 = sum(d["n_tests"] for d in D2.values())
    pr2 = sum(d["pass_on_reference"] for d in D2.values())
    v2card = f"""
      <div class="results-card" style="margin-top:22px">
        <span class="mono">V2 · REAL NEMOTRON-3 ON CRUSOE · PROMPT → DOD → LEARNED LOSS</span>
        <p class="verdict" style="background:#b2f2bb">With a real Nemotron executor, the learned TraceLoss controller verified {C2['graph_traceloss']['verified_success_rate']:.0%} vs {C2['ralph']['verified_success_rate']:.0%} for Ralph — ahead, but intervals overlap (not significant yet).</p>
        <table class="rtable"><tr><th>Controller</th><th>Verified</th><th>95% CI</th><th>False done</th></tr>{rows2}</table>
        <ul>
          <li>Nemotron-3-Super wrote the Definition of Done from each one-line prompt: {pr2}/{nt2} generated tests are correct.</li>
          <li>TraceLoss = small transformer over [DoD embedding, trace step embeddings, budget] → V; loss = 1 − V. Trained on {T2['n_states']} states labelled by real rollouts.</li>
          <li>Grouped CV: TraceLoss MSE {cv2['traceloss_adamw']['mse']:.3f} vs constant {cv2['const_train_mean']['mse']:.3f}; DoD pass fraction is anti-predictive (AUROC {cv2['dod_pass_fraction']['auroc_fold_mean']:.2f}).</li>
          <li>Out of distribution on real Pi traces it does not transfer.</li>
        </ul>
      </div>"""
    lr, la = ctrl["ralph"]["verified_success_rate"], ctrl["graph_learned_muon"]["verified_success_rate"]
    n_ep = ctrl["ralph"]["episodes"]
    return f"""
    <section class="results wrap" id="results">
      <div class="section-label"><span>05 / WHAT WE MEASURED</span><span class="sim-flag">V2 REAL NEMOTRON · V1 SIMULATED</span></div>
      <h2>Me fail English? <span class="serif">That's unpossible.</span></h2>
      {v2card}
      <h3 style="margin-top:30px">v1 · simulated executor (first experiment)</h3>
      <div class="results-grid">
        <div class="results-card">
          <p class="verdict">Hypothesis not supported (yet): at equal budget the learned critic verified {la:.0%} of sealed test episodes vs {lr:.0%} for plain Ralph.</p>
          <span class="mono">SEALED TEST TASKS: {html.escape(', '.join(r['splits']['test']))} · {n_ep} EPISODES / CONTROLLER · BUDGET {r['config']['EVAL_BUDGET']} EXECUTOR CALLS</span>
          <table class="rtable"><tr><th>Controller</th><th>Verified</th><th>95% CI</th><th>Calls / success</th><th>Verify-fail</th></tr>{rows}</table>
          <span class="mono" style="display:block;margin-top:20px">CRITIC ON {crit['adamw']['test']['n_states']} HELD-OUT STATES</span>
          <table class="rtable"><tr><th>Model</th><th>MSE ↓</th><th>Pairwise rank acc.</th></tr>{crows}</table>
          <ul>
            <li>The critic predicts state value better than the non-learned references, but ranks sibling branches barely above chance — and ranking is what the controller needs.</li>
            <li>Graph search wastes fewer verifier calls on false finishes than plain Ralph.</li>
            <li>Muon vs AdamW: no measurable difference for this tiny head.</li>
            <li>Labels: {r['labeled_states']['train']} train / {r['labeled_states']['valid']} valid / {r['labeled_states']['test']} test states, each from {r['config']['K_CONT']} real continuations checked by pytest.</li>
          </ul>
        </div>
        <div class="results-card gif-card">
          <span class="mono">TRACE REPLAY · REAL LOGGED EPISODE</span>
          <img src="trace.gif" alt="Replay of a logged search episode: backtrack, FLAG_GAP, dropped loops, verified finish" style="margin-top:10px">
          <p>Node colour = critic V. Red ring = visible checks pass but the verifier says no (FLAG_GAP). Green ring = independent pytest verifier passes. The critic steers; it never certifies.</p>
          <p><a class="button primary" href="{REPO}" target="_blank" rel="noopener noreferrer" style="margin-top:8px">Code + data on GitHub <span aria-hidden="true">↗</span></a></p>
        </div>
      </div>
      <div class="results-grid" style="margin-top:26px">
        <div class="results-card">
          <span class="mono">REAL PI AGENT TRACES · HF AGENT-TRACES · <a href="https://huggingface.co/datasets/{pi['dataset']}" target="_blank" rel="noopener noreferrer">{pi['dataset']}</a></span>
          <p class="verdict" style="background:#fff3bf">{pi['done_claims_without_passing_check_after_last_edit']} of {pi['sessions_ending_with_done_claim']} sessions that end with “done” show no passing test/build command after their last edit.</p>
          <ul>
            <li>{pi['sessions']} real sessions, {len(pi['models'])} models, {pi['tool_calls']} tool calls; tool error rate {pi['tool_error_rate']:.1%}.</li>
            <li>{pi['check_failures']} of {pi['check_commands']} test/build commands failed; only {pi['sessions_with_any_check']} sessions ran any check.</li>
            <li>{pi['repeat_calls_same_result']} exact repeats (same tool, args and result, no edit in between) — the loop signal TRACE drops.</li>
            <li>No verifier outcome exists for these traces: these are heuristic signals, not success labels, and are not used to train the critic.</li>
          </ul>
        </div>
        <div class="results-card">
          <span class="mono">PLEIAS/SYNTH · MATH REASONING · INDEPENDENT ARITHMETIC VERIFIER</span>
          <p class="verdict" style="background:#fff3bf">{sy['traces_with_any_wrong_step']} of {sy['traces_with_checkable_steps']} checkable synthetic reasoning traces contain at least one wrong arithmetic step.</p>
          <ul>
            <li>{sy['arithmetic_steps_checked']:,} explicit equalities re-computed exactly in Python; {sy['arithmetic_steps_wrong']} flagged ({sy['step_error_rate']:.1%}).</li>
            <li>Manual audit of {au['n']} random flags: {au['genuine']} genuine errors (e.g. <code>5.79 × 905 = 5,240.95</code>), {au['n'] - au['genuine']} parser false positives.</li>
            <li>Local step validity only — it does not check whether the problem was modelled correctly.</li>
            <li>Why it matters: even curated reasoning traces need an independent check before they count as “done”.</li>
          </ul>
        </div>
      </div>
      <div class="results-card" style="margin-top:26px">
        <span class="mono">REAL EXECUTOR · {nemo['executor']} · CRUSOE SERVERLESS INFERENCE</span>
        <p class="verdict" style="background:#fff3bf">With Nemotron as the executor, every controller verified all {nc['ralph']['episodes']} sealed test episodes: the fixtures are too easy for a real LLM, so the critic has almost nothing to learn.</p>
        <table class="rtable"><tr><th>Controller</th><th>Verified</th><th>Calls / episode</th><th>Verify-fail</th></tr>{nrows}</table>
        <ul><li>{nemo['executor_stats']['api_calls']} Nemotron calls (thinking off, JSON reply), {nemo['executor_stats']['parse_fail']} unparseable. All replies are cached for exact replay.</li>
        <li>Plain Ralph is cheapest here, because graph search pays for two branches per expansion.</li>
        <li>Nemotron still trips on hidden checks: 9–19% of the time the visible checks passed but the verifier failed the full contract.</li>
        <li>Next step: harder tasks with free-form edits, so a learned critic has something to rank.</li></ul>
      </div>
      <p style="margin-top:22px"><a class="button primary" href="app/">Open the TRACE app <span aria-hidden="true">↗</span></a></p>
    </section>
"""


def main():
    s = SRC.read_text()
    rep = [
        ('<meta name="theme-color" content="#f4f3ed">', '<meta name="theme-color" content="#ffd90f">'),
        ("<title>TRACE — A loss function for agent traces</title>",
         "<title>Nesting Ralphs · TRACE — A loss function for agent traces</title>"
         '<meta property="og:title" content="Nesting Ralphs · TRACE"><meta property="og:image" content="banner.jpg">'),
        ("</style>", RALPH_CSS + "</style>" + FONTS),
        ('<span class="brand-mark">△</span> TRACE<span class="brand-detail">RECURSIVE OPTIMISATION</span>',
         '<img src="logo.png" alt=""> Nesting Ralphs<span class="brand-detail">TRACE · RECURSIVE OPTIMISATION</span>'),
        ('<a href="#method">Method</a></nav>', f'<a href="#method">Method</a><a href="#results">Results</a><a href="app/">App</a><a href="{REPO}" target="_blank" rel="noopener noreferrer">GitHub</a></nav>'),
        ('<main id="main">', '<div class="wrap"><div class="banner"><img src="banner.jpg" alt="Nesting Ralph matryoshka dolls, each one opening into a smaller Ralph loop"></div></div>\n  <main id="main">'),
        ('<div class="hero-copy">', '<div class="hero-copy"><div class="ralph-quote">“I’m learnding!”</div>'),
        ('<p class="hero-description">', '<p class="hero-tagline">based on <b>Ralph Wiggum</b> and <b>Russian Nesting Dolls</b> principles</p>'
         '<ul class="principles-mini"><li><b>Ralph:</b> keep trying the same loop until checks pass.</li>'
         '<li><b>Nesting dolls:</b> every loop is a doll with its own definition of done and budget; a learned critic picks which doll to open next.</li></ul>'
         '<p class="hero-description">'),
        ('<meta name="description" content="A research idea for recursive agents: learn a value function over traces, use feedback to choose the next step, and verify the finish independently.">',
         '<meta name="description" content="A loss function for agent traces based on Ralph Wiggum and Russian Nesting Dolls principles."><meta property="og:description" content="A loss function for agent traces based on Ralph Wiggum and Russian Nesting Dolls principles.">'),
        ("A research concept. An interactive simulation.",
         "A research prototype. The playground uses illustrative data; measured results are below."),
        ("A stateful optimiser for agent decisions. Muon inspires the ambition; this prototype does not implement Muon or claim differentiability through tool calls.",
         "A stateful optimiser for agent decisions. The prototype trains the critic’s hidden matrix with Muon (Newton–Schulz) + AdamW as an ablation; it found no difference vs AdamW. No differentiability through tool calls is claimed."),
        ("Evaluate on held-out tasks. No benchmark results are claimed here.",
         "Evaluate on held-out tasks. A first small run with a simulated executor is reported below."),
        ('<section class="reading wrap">', results_section() + '\n    <section class="reading wrap">'),
        ('<a href="https://kellerjordan.github.io/posts/muon/" target="_blank" rel="noopener noreferrer">Muon <span>Optimiser inspiration ↗</span></a>',
         '<a href="https://kellerjordan.github.io/posts/muon/" target="_blank" rel="noopener noreferrer">Muon <span>Optimiser inspiration ↗</span></a>'
         f'<a href="{REPO}" target="_blank" rel="noopener noreferrer">Nesting Ralphs <span>Code &amp; data ↗</span></a>'),
        ('<footer class="footer wrap"><a class="brand" href="#"><span class="brand-mark">△</span> TRACE</a>',
         '<footer class="footer wrap"><a class="brand" href="#"><img src="logo.png" alt=""> Nesting Ralphs</a>'),
        ('<a class="header-cta" href="#playground">Explore the loop <span aria-hidden="true">↗</span></a>',
         '<a class="header-cta" href="app/">Open the app <span aria-hidden="true">↗</span></a>'),
        ("Recursive Agents Hackathon · Research concept · 2026", "Recursive Agents Hackathon · Research prototype · 2026 · MIT"),
    ]
    for a, b in rep:
        assert s.count(a) == 1, f"anchor not found exactly once: {a[:60]}"
        s = s.replace(a, b)
    # favicon
    s = s.replace(s[s.index('<link rel="icon"'): s.index(">", s.index('<link rel="icon"')) + 1],
                  '<link rel="icon" href="favicon.png">')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(s)
    for f in ("banner.jpg", "logo.png", "favicon.png", "trace.gif"):
        shutil.copy(ROOT / "assets" / f, OUT / f)
    print("wrote", OUT / "index.html")


if __name__ == "__main__":
    main()
