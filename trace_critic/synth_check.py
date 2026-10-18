"""Independent verifier for PleIAs/SYNTH 'math exercise' reasoning traces.

Source: https://huggingface.co/datasets/PleIAs/SYNTH (shard synth_001.parquet, exercise == 'math exercise').
Every explicit arithmetic equality in `synthetic_reasoning` (e.g. `178 × 90 = 16020`, also chains
`a = b = c`) is re-evaluated with exact rational arithmetic in Python. This gives LOCAL step-validity
labels. It does not check whether the problem was modelled correctly, so a trace with zero arithmetic
errors is not thereby "correct"; we report it as "no detected arithmetic error".

A final-answer consistency check was tried and dropped: extracting "the final number" from bold
markdown proved unreliable on manual inspection.

Usage: uv run python -m trace_critic.synth_check
"""

import ast
import json
import operator
import re
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/synth/math_exercise_synth_001.jsonl"
OUT = ROOT / "runs/synth_audit.json"

NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"
SKIP_LINE = re.compile(r"[%≈~√°⁰¹²³⁴⁵⁶⁷⁸⁹∛π⌊⌋⌈⌉·]|\.\.\.|…|\(\s*-?[\d.]+\s*,\s*-?[\d.]+|remainder|\brem\b|\bR\d|digit|mod\b|[A-Za-z]{2,}\s*\(|\b[a-z]\s*\(", re.I)
EXPR_CHARS = re.compile(r"^[\d\s.,()+\-*/^]+$")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def normalize(s):
    s = s.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    s = re.sub(r"(?<=\d)\s*[x]\s*(?=\d)", "*", s)  # 3 x 4
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)       # thousands separators
    s = s.replace("$", "").replace("€", "")
    return s


def evaluate(expr):
    tree = ast.parse(expr.replace("^", "**"), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return Fraction(str(n.value))
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow):
                if b.denominator != 1 or abs(b) > 12:
                    raise ValueError("pow")
                return a ** int(b)
            return OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS:
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported")
    return ev(tree)


def _side(seg, left):
    """Largest arithmetic-only suffix (left side) or prefix (right side) of a segment.
    Returns None if the expression is glued to a variable/word (e.g. 'x - 52', '3x + 4'),
    because then the visible numbers are not the whole expression."""
    seg = normalize(seg)
    m = re.search(r"[\d\s.()+\-*/^]+$", seg) if left else re.match(r"^[\d\s.()+\-*/^]+", seg)
    if not m:
        return None
    if left:
        before = seg[:m.start()].rstrip()
        if before and (before[-1].isalnum() or before[-1] in "_)]}'\\"):
            return None
        s = m.group(0).strip()
        s = re.sub(r"^[-+*]\s+", "", s)  # markdown bullet / continuation marker, not a sign
        if not before and s.startswith("-") and not re.match(r"^-\d", s):
            return None
    else:
        after = seg[m.end():].lstrip()
        if after and (after[0].isalpha() or after[0] in "_([{\\"):
            # e.g. '= 2x + 1' or '= 12(3)': value continues into a symbol
            if not re.match(r"^[A-Za-z]{2,}", after):  # allow units/words like 'pounds', 'apples'
                return None
        s = m.group(0)
    s = s.strip().strip(".").strip()
    # balance parentheses by trimming unmatched ones at the outer edge
    while s.count("(") > s.count(")") and s.startswith("("):
        s = s[1:].strip()
    while s.count(")") > s.count("(") and s.endswith(")"):
        s = s[:-1].strip()
    if not s or not re.search(r"\d", s) or s.count("(") != s.count(")"):
        return None
    return s


def decimals(s):
    m = re.findall(r"\.(\d+)", s)
    return max((len(x) for x in m), default=0)


def check_equalities(text):
    steps = []
    for line_no, line in enumerate(text.splitlines()):
        if "=" not in line or SKIP_LINE.search(line):
            continue
        segs = line.split("=")
        for i in range(len(segs) - 1):
            lhs, rhs = _side(segs[i], True), _side(segs[i + 1], False)
            if not lhs or not rhs:
                continue
            if not re.search(r"\d\s*[+\-*/^]\s*[\d(]", lhs + " " + rhs):
                continue  # need at least one operation, otherwise it is just a restatement
            try:
                a, b = evaluate(lhs), evaluate(rhs)
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError, TypeError):
                continue
            tol = Fraction(1, 2) * Fraction(1, 10 ** min(decimals(lhs), decimals(rhs))) if (decimals(lhs) or decimals(rhs)) else 0
            if "/" in lhs + rhs and not (decimals(lhs) or decimals(rhs)):
                tol = max(tol, Fraction(1, 2))  # integer answers to divisions are often rounded
            ok = abs(a - b) <= tol
            steps.append({"line": line_no, "expr": f"{lhs} = {rhs}", "ok": ok,
                          "lhs": float(a), "rhs": float(b)})
    return steps


def bold_numbers(text):
    return [normalize(x) for x in re.findall(r"\*\*[^*\d-]*(-?" + NUM + r")[^*]*\*\*", text)]


def main():
    rows = [json.loads(l) for l in open(SRC)]
    out, n_steps, n_bad, traces_bad, consist_n, consist_ok = [], 0, 0, 0, 0, 0
    first_bad_pos, corrected = [], 0
    for r in rows:
        steps = check_equalities(r["synthetic_reasoning"])
        bad = [s for s in steps if not s["ok"]]
        n_steps += len(steps)
        n_bad += len(bad)
        traces_bad += bool(bad)
        if bad:
            i = steps.index(bad[0])
            first_bad_pos.append(i / max(1, len(steps) - 1))
            corrected += steps[-1]["ok"]
        ra, aa = bold_numbers(r["synthetic_reasoning"]), bold_numbers(r["synthetic_answer"])
        consistent = None
        if ra and aa:
            consist_n += 1
            try:
                consistent = abs(float(ra[-1]) - float(aa[0])) < 1e-9
            except ValueError:
                consistent = None
            consist_ok += bool(consistent)
        out.append({"synth_id": r["synth_id"], "language": r["language"], "model": r["model"],
                    "query": r["query"], "steps": steps, "n_bad": len(bad),
                    "final_reasoning": ra[-1] if ra else None, "final_answer": aa[0] if aa else None,
                    "answer_consistent": consistent})
    with_steps = sum(1 for o in out if o["steps"])
    summary = {
        "dataset": "PleIAs/SYNTH", "shard": "synth_001.parquet", "exercise": "math exercise", "license": "CC-BY-SA 4.0 (per row: seed_license)",
        "traces": len(rows), "traces_with_checkable_steps": with_steps,
        "arithmetic_steps_checked": n_steps, "arithmetic_steps_wrong": n_bad,
        "step_error_rate": n_bad / max(1, n_steps),
        "traces_with_any_wrong_step": traces_bad,
        "trace_error_rate_among_checkable": traces_bad / max(1, with_steps),
        "wrong_traces_whose_last_step_is_correct": corrected,
        "mean_relative_position_of_first_error": sum(first_bad_pos) / max(1, len(first_bad_pos)),
        "final_answer_consistency": "not reported: bold-number extraction was unreliable on manual inspection",
        "caveat": ("Local arithmetic validity only: problem modelling is not verified. Parser covers explicit "
                   "'expr = value' equalities; percentages, approximations and prose arithmetic are skipped. "
                   "Flagged steps include possible parser false positives; inspect examples before citing."),
    }
    OUT.write_text(json.dumps({"summary": summary, "traces": out}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
