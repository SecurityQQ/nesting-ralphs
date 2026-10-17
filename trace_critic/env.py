"""Contract, isolated snapshots and the independent pytest verifier."""

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
VERIFIER_VERSION = "pytest-v1"


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Contract:
    task: str
    version: str
    criteria: tuple  # (criterion_id, mandatory, visible)
    tests_sha: str
    path: str


def contract_source(task):
    lines = ["import pytest", "import solution", ""]
    for cid, body in task.tests.items():
        lines += [f"def test_{cid}():", f"    {body}", ""]
    return "\n".join(lines)


_CONTRACTS = {}
_LOCK = __import__("threading").Lock()


def get_contract(task_name):
    with _LOCK:
        if task_name not in _CONTRACTS:
            _CONTRACTS[task_name] = build_contract(task_name)
        return _CONTRACTS[task_name]


def build_contract(task_name):
    """Freeze the contract for one episode. Tests live outside the executor's workspace."""
    task = TASKS[task_name]
    src = contract_source(task)
    d = RUNS / "contracts" / task_name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "test_contract.py"
    p.write_text(src)
    crit = tuple((c, True, c in task.visible) for c in task.criteria)
    c = Contract(task_name, "c1", crit, sha(src), str(p))
    (d / "contract.json").write_text(json.dumps(
        {"task": task_name, "goal": task.goal, "version": c.version, "tests_sha256_16": c.tests_sha,
         "verifier_version": VERIFIER_VERSION,
         "criteria": [{"criterion_id": a, "mandatory": b, "visible_to_executor": v, "check": f"pytest::test_{a}"}
                      for a, b, v in crit]}, indent=2))
    return c


_CACHE = {}


def verify(task_name, variant):
    """Run the frozen contract tests on the rendered artifact. Returns (artifact_hash, {criterion: bool}) or raises."""
    task = TASKS[task_name]
    src = task.render(variant)
    h = sha(src)
    key = (task_name, h)
    if key in _CACHE:
        return h, _CACHE[key]
    contract = get_contract(task_name)
    tests = Path(contract.path).read_text()
    assert sha(tests) == contract.tests_sha, "contract tampered"
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "solution.py").write_text(src)
        Path(tmp, "test_contract.py").write_text(tests)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rA", "--tb=no", "-p", "no:cacheprovider",
                            "test_contract.py"], cwd=tmp, capture_output=True, text=True, timeout=60)
    res = {}
    for m in re.finditer(r"^(PASSED|FAILED|ERROR) test_contract\.py::test_(\w+)", r.stdout, re.M):
        res[m.group(2)] = m.group(1) == "PASSED"
    if set(res) != set(task.criteria):
        raise RuntimeError(f"invalid_run for {task_name}: {r.stdout[-400:]}")
    _CACHE[key] = res
    return h, res


def all_variants(task):
    out = [()]
    for s in task.sites:
        out = [v + (i,) for v in out for i in range(len(s))]
    return out


def warm_cache(names):
    """Pre-run the verifier on every reachable artifact (deterministic; parallel)."""
    jobs = [(n, v) for n in names for v in all_variants(TASKS[n])]
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        list(ex.map(lambda j: verify(*j), jobs))
    return len(jobs)
