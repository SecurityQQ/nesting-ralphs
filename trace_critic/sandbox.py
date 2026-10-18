"""Run a pytest file against a solution source in an isolated temp dir."""

import hashlib
import re
import subprocess
import sys
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path

LINE = re.compile(r"^(PASSED|FAILED|ERROR) test_\w+\.py::(test_\w+)(?: - (.*))?$", re.M)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def tests_from_dict(d):
    out = ["import pytest", "import solution", ""]
    for cid, body in d.items():
        out += [f"def test_{cid}():", textwrap.indent(body, "    "), ""]
    return "\n".join(out)


@lru_cache(maxsize=50000)
def run_tests(src, test_code, timeout=20):
    """Returns (status, {test_name: (passed, message)}). status: ok | collect_error | timeout."""
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "solution.py").write_text(src)
        Path(tmp, "test_x.py").write_text(test_code)
        try:
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "--tb=line", "-p", "no:cacheprovider",
                                "-x" if False else "-q", "test_x.py"], cwd=tmp, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout", {}
    res = {m.group(2): (m.group(1) == "PASSED", (m.group(3) or "")[:200]) for m in LINE.finditer(r.stdout)}
    if not res:
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        return "collect_error", {"_error": (False, " | ".join(tail)[:300])}
    return "ok", res


def hidden_verify(task, src):
    status, res = run_tests(src, tests_from_dict(task.hidden))
    if status != "ok":
        return False, {}
    return all(ok for ok, _ in res.values()), {k[5:]: ok for k, (ok, _) in res.items()}
