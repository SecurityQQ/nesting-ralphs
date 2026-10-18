"""Nemotron (Crusoe Serverless Inference): prompt -> Definition of Done, and a free-form Ralph executor.

The DoD is generated from the one-line goal + buggy source only. The hidden tests are never shown
to the DoD generator or the executor; they are the independent verifier.
"""

import asyncio
import difflib
import json
import os
import re
from pathlib import Path

from dotenv import dotenv_values
from openai import AsyncOpenAI

from .sandbox import hidden_verify, run_tests, sha

ROOT = Path(__file__).resolve().parent.parent
_env = {**dotenv_values(ROOT / ".env"), **os.environ}
BASE_URL = _env.get("CRUSOE_INFERENCE_BASE_URL", "https://api.inference.crusoecloud.com/v1")
DOD_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"
EXEC_MODEL = _env.get("TRACE_EXEC_MODEL", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B")
_client = None
SEM = asyncio.Semaphore(int(_env.get("TRACE_CONCURRENCY", "32")))
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "errors": 0}
CODE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
BAD_TEST = re.compile(r"NameError|ImportError|ModuleNotFoundError|SyntaxError|fixture '|IndentationError")


def client():
    global _client
    if _client is None:
        _client = AsyncOpenAI(base_url=BASE_URL, api_key=_env["CRUSOE_INFERENCE_API_KEY"], timeout=120, max_retries=4)
    return _client


async def chat(model, prompt, temperature=0.7, max_tokens=2500):
    async with SEM:
        try:
            r = await client().chat.completions.create(model=model, temperature=temperature, max_tokens=max_tokens,
                                                       messages=[{"role": "user", "content": prompt}])
        except Exception as e:  # infra failure: counted, surfaced to the caller as empty output
            USAGE["errors"] += 1
            return "", str(e)[:200]
    USAGE["calls"] += 1
    if r.usage:
        USAGE["prompt_tokens"] += r.usage.prompt_tokens or 0
        USAGE["completion_tokens"] += r.usage.completion_tokens or 0
    return r.choices[0].message.content or "", None


def last_code(text):
    blocks = CODE.findall(text or "")
    return blocks[-1] if blocks else None


# ---------------------------------------------------------------- Definition of Done
DOD_PROMPT = """You write the Definition of Done for a coding task as executable pytest tests.

Task (one-line prompt): {goal}

Current (buggy) solution.py:
```python
{src}
```

Write 5 to 8 pytest test functions that together define when the task is DONE:
- one test per requirement stated or clearly implied by the task, including edge cases and error cases;
- import the code with `import solution` and call `solution.{func}(...)`; use `pytest.raises` for errors;
- name tests `test_<criterion>` and give each a one-line docstring stating the requirement;
- test behaviour only, never implementation details; expected values must follow from the task text.

Reply with a single ```python block containing the complete test file."""


async def generate_dod(task, attempts=3):
    feedback = ""
    for a in range(attempts):
        out, err = await chat(DOD_MODEL, DOD_PROMPT.format(goal=task.goal, src=task.buggy, func=task.func) + feedback,
                              temperature=0.3)
        code = last_code(out)
        if not code:
            feedback = "\n\nYour previous reply had no python code block."
            continue
        if "import pytest" not in code:
            code = "import pytest\n" + code
        status, res = await asyncio.to_thread(run_tests, task.buggy, code)
        if status != "ok":
            feedback = f"\n\nYour previous test file failed to collect: {res.get('_error', ('', ''))[1]}. Fix it."
            continue
        # Auditor: drop tests whose failure is a bug in the test itself, keep the rest.
        broken = [n for n, (ok, msg) in res.items() if not ok and BAD_TEST.search(msg)]
        code = drop_tests(code, broken)
        status, res = await asyncio.to_thread(run_tests, task.buggy, code)
        n_fail = sum(not ok for ok, _ in res.values())
        if status == "ok" and len(res) >= 3 and n_fail >= 1:
            ref_status, ref = await asyncio.to_thread(run_tests, task.reference, code)
            return {"task": task.name, "model": DOD_MODEL, "attempt": a + 1, "code": code, "tests": sorted(res),
                    "dropped_by_auditor": broken, "fail_on_buggy": n_fail, "n_tests": len(res),
                    # quality only, never used by the agent or the controller:
                    "pass_on_reference": sum(ok for ok, _ in ref.values()) if ref_status == "ok" else 0}
        feedback = "\n\nYour previous tests did not discriminate the buggy code (no failing test). Be stricter."
    return None


def drop_tests(code, names):
    for n in names:
        code = re.sub(rf"\ndef {n}\(.*?(?=\ndef |\Z)", "\n", code, flags=re.S)
    return code


def dod_summary(dod):
    """Criteria text for embedding: test names + docstrings."""
    items = re.findall(r'def (test_\w+)\(.*?\):\s*(?:"""(.*?)""")?', dod["code"], re.S)
    return "; ".join(f"{n[5:]}: {(d or '').strip()}" for n, d in items)


# ---------------------------------------------------------------- executor
STEP_PROMPT = """You are an autonomous coding agent fixing solution.py.

Task: {goal}

Definition of Done (you are done when all these pytest tests pass):
```python
{dod}
```

Current solution.py:
```python
{src}
```

Last test run: {passed}/{total} DoD tests passed.{failing}{history}{feedback}

Make ONE focused change in this step (fix a single issue). Reply exactly in this format:
THOUGHT: <one sentence about what you change and why>
```python
<the complete new solution.py>
```"""


def run_dod(src, dod):
    status, res = run_tests(src, dod["code"])
    if status != "ok":
        return {"passed": 0, "total": dod["n_tests"], "failing": [["_collect", str(res.get("_error", ("", status))[1])[:150]]]}
    return {"passed": sum(ok for ok, _ in res.values()), "total": len(res),
            "failing": [[n[5:], m[:150]] for n, (ok, m) in sorted(res.items()) if not ok]}


def step_text(step):
    """What gets embedded for one trace step: thought, code change, observation."""
    d = step["dod"]
    parts = [f"THOUGHT: {step.get('thought', '')}"]
    if step.get("diff"):
        parts.append("CHANGE:\n" + step["diff"][:700])
    parts.append(f"DOD: {d['passed']}/{d['total']} passed." +
                 (" FAILING: " + "; ".join(f"{n}: {m}" for n, m in d["failing"][:4]) if d["failing"] else ""))
    if step.get("verified") is False:
        parts.append("INDEPENDENT VERIFIER: REJECTED")
    return "\n".join(parts)


def initial_state(task, dod):
    d = run_dod(task.buggy, dod)
    return {"src": task.buggy, "verify_fails": 0,
            "steps": [{"i": 0, "thought": "initial buggy code", "diff": "", "dod": d, "artifact": sha(task.buggy),
                       "src_snapshot": task.buggy,
                       "verified": None, "kind": "ROOT"}]}


async def step(task, dod, state, temperature=0.8):
    last = state["steps"][-1]
    d = last["dod"]
    failing = ("\nFailing tests:\n" + "\n".join(f"- {n}: {m}" for n, m in d["failing"][:6])) if d["failing"] else ""
    hist = [s["thought"] for s in state["steps"][1:]][-3:]
    history = ("\nYour previous steps:\n" + "\n".join(f"- {h}" for h in hist)) if hist else ""
    feedback = ("\nIMPORTANT: an independent reviewer REJECTED the last version although all DoD tests pass. "
                "Re-read the task for requirements the tests miss.") if last.get("verified") is False else ""
    out, err = await chat(EXEC_MODEL, STEP_PROMPT.format(goal=task.goal, dod=dod["code"], src=state["src"],
                                                         passed=d["passed"], total=d["total"], failing=failing,
                                                         history=history, feedback=feedback), temperature=temperature)
    m = re.search(r"THOUGHT:\s*(.+)", out or "")
    thought = (m.group(1).strip() if m else (err or "no thought"))[:300]
    code = last_code(out)
    new_src = code if code and code.strip() else state["src"]
    diff = "\n".join(l for l in difflib.unified_diff(state["src"].splitlines(), new_src.splitlines(), lineterm="", n=0)
                     if l[:1] in "+-" and not l.startswith(("+++", "---")))
    nd = await asyncio.to_thread(run_dod, new_src, dod)
    rec = {"i": len(state["steps"]), "thought": thought, "diff": diff, "dod": nd, "artifact": sha(new_src),
           "src_snapshot": new_src,
           "verified": None, "kind": "STEP" if code else "NO_CODE", "infra_error": bool(err)}
    new = {"src": new_src, "verify_fails": state["verify_fails"], "steps": state["steps"] + [rec]}
    return new


async def maybe_verify(task, state):
    """Done is claimed when all DoD tests pass; the independent hidden verifier decides."""
    rec = state["steps"][-1]
    if rec["dod"]["passed"] == rec["dod"]["total"] and rec["dod"]["total"] > 0:
        ok, _ = await asyncio.to_thread(hidden_verify, task, state["src"])
        rec["verified"] = ok
        if not ok:
            state["verify_fails"] += 1
        return ok, True
    return False, False


async def ralph(task, dod, state, budget, temperature=0.8):
    """Bounded Ralph: same loop until verified or budget exhausted.
    Returns (success, calls, verify_calls, final_state)."""
    calls = vcalls = 0
    ok, did = await maybe_verify(task, state)
    vcalls += did
    if ok:
        return True, 0, vcalls, state
    while calls < budget:
        state = await step(task, dod, state, temperature)
        calls += 1
        ok, did = await maybe_verify(task, state)
        vcalls += did
        if ok:
            return True, calls, vcalls, state
    return False, calls, vcalls, state


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
