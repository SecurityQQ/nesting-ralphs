"""Import real Pi Agent sessions (Hugging Face `format:agent-traces`) and audit them with
the same environment-only signals the TRACE controller uses.

Source: https://huggingface.co/datasets/championswimmer/pi-coding-sessions (MIT).
These traces have NO verifier outcome. Every label here is a heuristic detector signal,
not a success/failure label, and is never used as a critic training target.

Usage: uv run python -m trace_critic.pi_traces
"""

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/pi"
OUT = ROOT / "runs/pi_audit.json"
DATASET = "championswimmer/pi-coding-sessions"

EDIT_TOOLS = {"edit", "write"}
CHECK_RE = re.compile(r"\b(pytest|test|tests|gradlew?\b.*\b(test|check|build|assemble)|npm (run )?(test|build|lint)|"
                      r"pnpm (run )?(test|build|lint)|yarn (test|build)|bun (test|run build)|cargo (test|build|check)|"
                      r"go (test|build|vet)|tsc|eslint|ruff|mypy|make( |$)|xcodebuild)\b")
DONE_RE = re.compile(r"^\s*(done|all done|finished|completed|implemented|fixed)\b|all (tests|checks) pass", re.I | re.M)


def _h(x):
    return hashlib.sha1(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _text(content):
    if isinstance(content, str):
        return content
    return "\n".join(c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text")


def load_session(path):
    lines = [json.loads(l) for l in open(path) if l.strip()]
    head = next((x for x in lines if x.get("type") == "session"), {})
    model = next((x.get("modelId") for x in lines if x.get("type") == "model_change"), None)
    calls, results, user_msgs, final_text = {}, {}, 0, ""
    order = []
    for x in lines:
        if x.get("type") != "message":
            continue
        m = x["message"]
        if m["role"] == "user":
            user_msgs += 1
            order.append(("user", None))
        elif m["role"] == "assistant":
            for c in m.get("content", []):
                if isinstance(c, dict) and c.get("type") == "toolCall":
                    calls[c["id"]] = c
                    order.append(("call", c["id"]))
            t = _text(m.get("content", []))
            if t.strip():
                final_text = t
                order.append(("text", t))
        elif m["role"] == "toolResult":
            results[m.get("toolCallId")] = m
    return head, model, calls, results, order, user_msgs, final_text


def audit(path):
    head, model, calls, results, order, user_msgs, final_text = load_session(path)
    events, seen = [], {}
    last_edit_step = last_check = None
    err_streak = max_err_streak = 0
    n = loops = errors = edits = checks = checks_failed = 0
    for kind, ref in order:
        if kind == "user":
            seen.clear()  # a new user instruction is new evidence; do not count repeats across it
            events.append({"step": len(events), "decision": "USER", "tool": None})
            continue
        if kind == "text":
            claim = bool(DONE_RE.search(ref))
            if claim:
                verified_after_edit = (last_check is not None and last_check["ok"] and
                                       (last_edit_step is None or last_check["step"] > last_edit_step))
                events.append({"step": len(events), "decision": "CLAIM_DONE" if verified_after_edit else "FLAG_GAP",
                               "tool": None, "detail": ref.strip()[:160]})
            continue
        c = calls[ref]
        r = results.get(ref, {})
        n += 1
        args = c.get("arguments") or {}
        out = _text(r.get("content", ""))
        is_err = bool(r.get("isError"))
        sig = (c.get("name"), _h(args))
        res_h = _h(out[:4000])
        decision = "CONTINUE"
        if c.get("name") in EDIT_TOOLS:
            edits += 1
            last_edit_step = len(events)
            seen.clear()  # an edit changes the environment: earlier observations may no longer hold
        elif sig in seen and seen[sig] == res_h:
            loops += 1
            decision = "REPLAN(loop)"
        seen[sig] = res_h
        if is_err:
            errors += 1
            err_streak += 1
            max_err_streak = max(max_err_streak, err_streak)
            if decision == "CONTINUE":
                decision = "TOOL_ERROR"
        else:
            err_streak = 0
        cmd = args.get("command", "") if c.get("name") == "bash" else ""
        is_check = bool(cmd) and bool(CHECK_RE.search(cmd))
        if is_check:
            checks += 1
            ok = not is_err
            checks_failed += not ok
            last_check = {"step": len(events), "ok": ok}
            decision = "CHECK_PASS" if ok else "CHECK_FAIL"
        events.append({"step": len(events), "decision": decision, "tool": c.get("name"),
                       "args": (cmd or json.dumps(args))[:200], "error": is_err,
                       "result": out[:300]})
    final_claim = bool(DONE_RE.search(final_text or ""))
    unverified_finish = final_claim and not (last_check and last_check["ok"] and
                                             (last_edit_step is None or last_check["step"] > last_edit_step))
    return {
        "session": head.get("id"), "file": path.name, "model": model, "user_messages": user_msgs,
        "tool_calls": n, "edits": edits, "tool_errors": errors, "max_error_streak": max_err_streak,
        "repeat_calls_same_result": loops, "check_commands": checks, "check_failures": checks_failed,
        "final_done_claim": final_claim, "done_claim_without_passing_check_after_last_edit": unverified_finish,
        "events": events,
    }


def main():
    sessions = [audit(p) for p in sorted(SRC.glob("*.jsonl"))]
    sessions = [s for s in sessions if s["tool_calls"] > 0]
    tot = lambda k: sum(s[k] for s in sessions)
    claims = [s for s in sessions if s["final_done_claim"]]
    summary = {
        "dataset": DATASET, "license": "MIT", "sessions": len(sessions),
        "models": sorted({s["model"] for s in sessions if s["model"]}),
        "tool_calls": tot("tool_calls"), "edits": tot("edits"), "tool_errors": tot("tool_errors"),
        "tool_error_rate": tot("tool_errors") / max(1, tot("tool_calls")),
        "repeat_calls_same_result": tot("repeat_calls_same_result"),
        "repeat_rate": tot("repeat_calls_same_result") / max(1, tot("tool_calls")),
        "check_commands": tot("check_commands"), "check_failures": tot("check_failures"),
        "sessions_with_any_check": sum(s["check_commands"] > 0 for s in sessions),
        "sessions_ending_with_done_claim": len(claims),
        "done_claims_without_passing_check_after_last_edit": sum(s["done_claim_without_passing_check_after_last_edit"] for s in claims),
        "caveat": ("Heuristic detectors over real Pi sessions; no independent verifier outcome exists for these traces. "
                   "'Done claim without passing check' means the session never showed a passing test/build command "
                   "after its last edit, not that the work was wrong."),
    }
    OUT.write_text(json.dumps({"summary": summary, "sessions": sessions}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
