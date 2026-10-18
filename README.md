<p align="center"><img src="assets/banner.jpg" alt="Nesting Ralphs banner" width="100%"></p>

<h1 align="center"><img src="assets/logo.png" alt="" width="96" align="center"> Nesting Ralphs</h1>

**A loss function for agent traces based on Ralph Wiggum and Russian Nesting Dolls principles.**

Ralph: keep trying the same loop until checks pass. Nesting dolls: every loop is a doll with its own definition of done and budget; a learned critic picks which doll to open next. The critic steers; it never certifies.

Live site: https://nesting-ralphs.vercel.app

A Ralph-style executor proposes and executes steps. A controller keeps checkpoints, compares
branches and reallocates attempts. A small critic `V(state, budget) ≈ P(verified success within budget)`
is trained on real rollout outcomes. An independent pytest verifier on a frozen contract decides success.
TRACE (`runs/trace.html`) replays the search history.

> Status: small-scale mechanism check. The main results table uses a seeded **simulated** executor. A second run with **NVIDIA Nemotron on Crusoe** is reported in its own section below.
> All numbers below are copied from `runs/results.json`, which `uv run python -m trace_critic.run` regenerates
> deterministically (≈6 s on a laptop).

<p align="center"><img src="assets/trace.gif" alt="TRACE replay of a real search episode" width="100%"></p>

<sub>Replay of a real logged episode (`graph_learned_muon` on the sealed `median` task, simulated executor):
the critic backtracks, a visibly-passing branch is flagged by the verifier (`FLAG_GAP`), repeats are dropped,
and the run ends only when the independent pytest verifier passes. Regenerate: `uv run python scripts/make_gif.py`.</sub>

## Quick start

```bash
uv sync
uv run python -m trace_critic.run            # simulated executor (default)
open runs/trace.html
```

Optional, with an OpenAI-compatible Nemotron endpoint:

```bash
export CRUSOE_INFERENCE_BASE_URL=... CRUSOE_INFERENCE_API_KEY=... CRUSOE_INFERENCE_MODEL=...
uv run python -m trace_critic.run --executor nemotron
```

## What is implemented

| Piece | File | Notes |
|---|---|---|
| 8 buggy fixture tasks, task-family split | `tasks.py` | train: 4 · valid: 1 (epoch/feature selection only) · test: 3 (sealed) |
| Frozen contract + verifier | `env.py` | `criterion_id → pytest::test_*`, SHA-256 of the tests checked on every call; tests live outside the executor workspace; executor sees only *visible* checks, verifier runs all (incl. hidden) |
| Executors | `executor.py` | `SimulatedExecutor` (seeded: follows failing checks, makes wrong edits, repeats itself); `NemotronExecutor` (OpenAI-compatible chat API) |
| Rollout labels | `run.py:collect` | states at action boundaries from 25 source trajectories/task; `K=6` continuations with budget `B ∈ {4, 8, 16}`; label `q = successes / valid_rollouts`, numerator and denominator stored |
| Critic | `critic.py` | MLP, `L = mean (V − q)²`; ablation **AdamW** vs **Muon (Newton–Schulz, hidden 2-D matrix) + AdamW** |
| Controllers | `search.py` | bounded Ralph · graph best-first with fixed rules · same graph search with learned critic; `K=2`, shared budget ledger, every executed branch charged |
| Decisions | `search.py` | `CONTINUE`, `BACKTRACK`, `REPLAN(loop)`, `FLAG_GAP` (verifier failed on a visibly-passing state), `VERIFIED`, `STOP_UNRESOLVED` |
| TRACE | `report.py`, `runs/traces/*.jsonl`, `site/public/app/` | HTML report, PWA replay app, export to HF Session Trace Simple Format |
| Real traces | `pi_traces.py`, `synth_check.py` | HF Agent Traces (Pi) import + audit; SYNTH arithmetic verifier |

Features are environment-only: visible-check status, deltas, repeats, budget, edit stats.
They exclude the task id, the narration, hidden checks and the outcome.

## Results (simulated executor, sealed test tasks, 180 episodes/controller, budget 16 executor calls)

| Controller | Verified success | 95% CI (Wilson) | Calls / verified success | Verify-fail rate |
|---|---|---|---|---|
| Bounded Ralph | 0.767 | [0.700, 0.822] | 12.35 | 0.622 |
| Graph + rules | 0.761 | [0.694, 0.818] | 12.80 | 0.372 |
| Graph + learned critic (AdamW) | 0.622 | [0.550, 0.690] | 16.32 | 0.429 |
| Graph + learned critic (Muon+AdamW) | 0.628 | [0.555, 0.695] | 16.34 | 0.423 |

Critic quality on 563 held-out test states:

| Model | MSE | Pairwise rank acc. |
|---|---|---|
| Critic, AdamW | 0.0714 | 0.550 |
| Critic, Muon+AdamW | 0.0718 | 0.563 |
| Reference: visible pass fraction | 0.1368 | 0.435 |
| Reference: constant train mean | 0.1178 | — |

### Honest reading

- **The hypothesis is not supported by this run.** At equal budget the learned controller verified *fewer* tasks than
  bounded Ralph or the rule-based graph search. Better state-value MSE did not translate into better branch selection.
- The critic generalizes to unseen task families better than the non-learned references. Its pairwise ranking is
  only slightly above chance, though, and ranking is what the controller needs.
- Graph search (rules or learned) roughly halves wasted verifications (verify-fail rate 0.37–0.43 vs 0.62).
- Muon vs AdamW: no measurable difference for this tiny head.
- Two fixes were made **before** reading test numbers again, and both are recorded in the code:
  1. Graph search used to drop expanded checkpoints, so it ran out of frontier at ≈7/16 calls.
     Checkpoints can now be re-expanded, with a priority penalty.
  2. Hashed source tokens acted as a task-identity proxy and destroyed generalization. The feature set
     (full vs structural) is now chosen on the **valid** family only (`feature_selection_valid_mse` in results).
- Likely next steps: a pairwise/ranking loss between sibling branches, labels from the *same* controller that
  consumes them (labels here come from Ralph continuations, i.e. a different policy), and a real LLM executor.

## Real executor: NVIDIA Nemotron on Crusoe

```bash
echo "CRUSOE_INFERENCE_API_KEY=..." >> .env      # Intelligence API key (not the Cloud API key)
uv run python -m trace_critic.run --executor nemotron   # -> runs/nemotron/
```

The executor is `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` on Crusoe Serverless Inference (`api.inference.crusoecloud.com/v1`), with thinking disabled.
With thinking on, the model spent its whole token budget reasoning and returned no answer.
Each proposal is one call. The model sees the goal, the source, the **visible** check results and the edit options, and replies with JSON `{"site", "option"}`.
It never sees hidden checks, critic scores or the verifier. Replies are cached in `runs/nemotron_cache.jsonl`, so the run replays exactly.
The run made 1012 API calls; 2 replies were unparseable and counted as charged no-ops. Config: `NEMOTRON_CFG` in `run.py`.

| Controller (Nemotron, 30 sealed test episodes each, budget 16) | Verified success | 95% CI | Executor calls / episode | Verify-fail rate |
|---|---|---|---|---|
| Bounded Ralph | 1.000 | [0.886, 1.000] | 3.60 | 0.143 |
| Graph + rules | 1.000 | [0.886, 1.000] | 4.87 | 0.189 |
| Graph + critic (AdamW) | 1.000 | [0.886, 1.000] | 5.53 | 0.189 |
| Graph + critic (Muon+AdamW) | 1.000 | [0.886, 1.000] | 5.27 | 0.091 |

**Honest reading: these fixtures are too easy for Nemotron.**
- Every controller verified every episode.
- 86 of 97 labeled states have label q = 1.0, so the critic has almost nothing to learn: a constant predictor matches it.
- Plain Ralph is the cheapest, because graph search pays for K=2 branches per expansion.
- Nemotron still gets caught by hidden checks: the verify-fail rate is 9–19%, meaning visible checks passed but the full contract did not.
- Testing the hypothesis with a real LLM needs harder tasks with free-form edits. That is the next step.

## v2: prompt → Definition of Done → learned loss (real Nemotron runs)

```
one-line prompt + code ──Nemotron-3-Super──▶ DoD (pytest tests, audited)
trace: step₁…step_t ──bge-small──▶ [z_DoD, z₁…z_t, budget] ──TraceLoss (2-layer transformer)──▶ V
loss = 1 − V        controller picks the branch with the highest V        hidden tests verify
```

```bash
PYTHONPATH=. uv run python scripts/pilot_v2.py          # DoD generation + pilot
PYTHONPATH=. uv run python -m trace_critic.collect_v2   # real trajectories + rollout labels
PYTHONPATH=. uv run python -m trace_critic.traceloss    # train + grouped CV
PYTHONPATH=. uv run python -m trace_critic.eval_v2      # controller comparison
```

Needs `CRUSOE_INFERENCE_API_KEY` in `.env` (Crusoe Serverless Inference).

- **Tasks:** 20 one-line prompts over buggy Python modules (`trace_critic/tasks_v2.py`).
  - Each task has hidden tests that the agent and the DoD generator never see; they are the independent verifier.
  - Each task also has a reference solution, used only to sanity-check the hidden tests and to score DoD quality.
- **DoD from the prompt:** `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B` writes the tests.
  - 20/20 prompts produced a usable DoD.
  - 135/141 generated tests are correct, i.e. they pass on the reference solution.
  - In 6 tasks (parse_duration, slugify, merge_intervals, luhn, word_wrap, csv_line) at least one test is wrong. A correct solution cannot pass such a DoD, so the agent either never claims done or has to break the solution to satisfy it.
- **Executor:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` makes free-form whole-file edits, one focused change per step.
- **Labels:** 215 trace states. Each state gets K=3 real continuations, and the label is q = the fraction verified by hidden tests. Collecting them took 1,596 Nemotron calls.
- **TraceLoss:** a 2-layer transformer over frozen `BAAI/bge-small-en-v1.5` embeddings, trained with MSE (V − q)².
  - The number of epochs is chosen by nested grouped CV.
  - The model is evaluated with 5-fold CV grouped by task.

**TraceLoss quality** (out-of-fold, 20 tasks):

| Model | MSE ↓ | Within-task rank ↑ | AUROC ↑ |
|---|---|---|---|
| TraceLoss (AdamW) | 0.135 | 0.625 | 0.604 |
| TraceLoss (Muon) | 0.116 | 0.605 | 0.608 |
| MLP on last step only | 0.422 | 0.718 | 0.458 |
| Constant | 0.160 | 0.000 | 0.500 |
| DoD pass fraction | 0.498 | 0.166 | 0.202 |
| Cosine(step, DoD), raw embedding heuristic | 0.591 | 0.372 | 0.710 |

On the sealed test tasks, the final model (trained on train+valid only) scores MSE 0.088. The constant scores 0.109 and DoD pass fraction scores 0.446.

**Controllers:** real Nemotron executor on the sealed test tasks (`roman_to_int, version_compare, flatten, word_wrap, median, csv_line`), budget 6 calls, 8 seeds per task.

| Controller | Verified | 95% CI | Calls / success | False done |
|---|---|---|---|---|
| Bounded Ralph | 0.667 | [0.53, 0.78] | 4.56 | 13/45 |
| Graph + rules (DoD pass fraction) | 0.688 | [0.55, 0.80] | 5.21 | 11/44 |
| Graph + TraceLoss (learned) | 0.729 | [0.59, 0.83] | 4.91 | 10/45 |

**Honest reading:**
- The learned loss is ahead, but the intervals overlap heavily, so the difference is not significant at this scale.
- The share of passing generated DoD tests is *anti-predictive* of verified success (AUROC 0.20). This is the false-done problem in numbers.
- Raw embedding similarity to the DoD ranks well (AUROC 0.71) but is badly calibrated. The learned head is calibrated but has smaller rank gains.
- Applied out of distribution to real Pi traces, TraceLoss collapses (V ≈ 0.04 everywhere): it does not transfer (`runs/v2/pi_curves.json`).
- The test split was re-drawn after a 60-episode pilot so that it is not at ceiling; pilot outcomes were not used otherwise.

## Real data: HF Agent Traces and PleIAs/SYNTH

Both sources are used as the spec intended: for import, auditing and verification. Neither is used as critic labels.

```bash
uv run python scripts/fetch_data.py        # downloads into data/ (gitignored)
uv run python -m trace_critic.pi_traces    # -> runs/pi_audit.json
uv run python -m trace_critic.synth_check  # -> runs/synth_audit.json
```

**Real Pi Agent sessions** ([championswimmer/pi-coding-sessions](https://huggingface.co/datasets/championswimmer/pi-coding-sessions), MIT, HF `format:agent-traces`).
These are imported with the controller's environment-only signals: exact repeats, tool errors, test/build commands,
and "done" claims with no passing check after the last edit.
26 sessions from 5 models, 1,695 tool calls, tool error rate 8.0%. 35 of 104 test/build commands failed, and only 13 sessions ran any check.
**9 of 15 sessions that end with "done" show no passing test/build command after their last edit.**
These traces have no verifier outcome, so these are heuristic signals, not success labels.

**PleIAs/SYNTH** (shard 1, `exercise == "math exercise"`, 2,443 traces). Every explicit equality in `synthetic_reasoning`
is recomputed exactly with `fractions.Fraction`. 12,385 equalities were checked and 285 flagged (2.3%).
**170 of 2,099 checkable traces contain at least one wrong arithmetic step.**
A manual audit of 30 random flags (`runs/synth_manual_audit.json`) found 27 genuine errors, e.g. `5.79 × 905 = 5,240.95`,
and 3 parser false positives. This checks local step validity only, not whether the problem was modelled correctly.
A final-answer consistency check was tried and dropped: its extraction proved unreliable.

## App

https://nesting-ralphs.vercel.app/app/ is an installable PWA with three tabs:
1. Step-by-step replay of logged controller episodes, with the search tree, critic V, code diff and visible vs hidden checks.
2. Real Pi Agent traces with detector signals.
3. SYNTH reasoning with verified and flagged arithmetic steps highlighted.

Rebuild data: `uv run python scripts/build_app.py`.

## Nemotron / Crusoe

Nemotron runs on Crusoe Serverless Inference (`https://api.inference.crusoecloud.com/v1`) through an Intelligence API key stored in `.env`, which is gitignored. The v2 numbers above all come from real Nemotron-3 calls. Raw logs are in `runs/v2/`.

## Not claimed

No novelty claims. This is a simulation. Value/process critics from Monte Carlo outcomes are prior work (e.g. AgentPRM).
Names and ideas borrowed from the "Matryoshka Ralph" spec (`FLAG_GAP`, "the loss steers, it never certifies") are
design choices; none of that spec's figures are reported here.

## License

MIT
