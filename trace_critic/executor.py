"""Executors propose actions. An action sets one editable site to one variant.

SimulatedExecutor: a seeded stochastic policy (NOT an LLM). It reads failing visible
checks, sometimes edits the hinted site, sometimes picks a wrong variant, and sometimes
repeats its last action (loops). Used for this run because no inference endpoint was verified.

NemotronExecutor: OpenAI-compatible chat endpoint (e.g. Crusoe Managed Inference).
Enabled with CRUSOE_INFERENCE_BASE_URL, CRUSOE_INFERENCE_API_KEY, CRUSOE_INFERENCE_MODEL.
"""

import json
import os
import random

from .tasks import TASKS


class SimulatedExecutor:
    policy_id = "sim-policy-v1"

    def __init__(self, p_hint=0.6, p_correct=0.45, p_repeat=0.2):
        self.p_hint, self.p_correct, self.p_repeat = p_hint, p_correct, p_repeat

    def propose(self, node, rng: random.Random):
        task = TASKS[node.task]
        if node.last_action and rng.random() < self.p_repeat:
            return node.last_action
        failing = [c for c, ok in node.observed.items() if not ok]
        hinted = [task.hints[c] for c in failing if c in task.hints]
        if hinted and rng.random() < self.p_hint:
            site = rng.choice(hinted)
        else:
            site = rng.randrange(len(task.sites))
        n = len(task.sites[site])
        if rng.random() < self.p_correct:
            v = 1
        else:
            v = rng.choice([i for i in range(n) if i != 1])
        return (site, v)


class NemotronExecutor:
    """NVIDIA Nemotron on Crusoe Serverless Inference (OpenAI-compatible API).

    Every proposal is one chat completion. Responses are cached on disk keyed by
    (model, prompt, seed) so a run can be replayed exactly and re-run for free.
    The executor sees the goal, the current source, the VISIBLE check results and the
    editable options; it never sees hidden checks, the critic score or the verifier.
    An unparseable or out-of-range reply becomes a no-op edit: the call is still charged.
    """

    DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B"

    def __init__(self, model=None):
        import threading
        from collections import Counter
        from pathlib import Path

        from openai import OpenAI
        root = Path(__file__).resolve().parent.parent
        envf = root / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())
        self.client = OpenAI(base_url=os.environ.get("CRUSOE_INFERENCE_BASE_URL", "https://api.inference.crusoecloud.com/v1"),
                             api_key=os.environ["CRUSOE_INFERENCE_API_KEY"], timeout=90, max_retries=6)
        self.model = model or os.environ.get("CRUSOE_INFERENCE_MODEL", self.DEFAULT_MODEL)
        self.policy_id = f"nemotron:{self.model}"
        self.cache_path = root / "runs" / "nemotron_cache.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                d = json.loads(line)
                self.cache[d["key"]] = d["reply"]
        self.lock = threading.Lock()
        self.stats = Counter()

    def prompt(self, node):
        task = TASKS[node.task]
        opts = "\n".join(
            f"site {i} (currently option {node.variant[i]}):\n" + "\n".join(f"  [{j}] {v!r}" for j, v in enumerate(s))
            for i, s in enumerate(task.sites))
        checks = "\n".join(f"  {c}: {'PASS' if ok else 'FAIL'}" for c, ok in node.observed.items())
        return (f"You are fixing a bug in a small Python module.\nGoal: {task.goal}\n\n"
                f"Current solution.py:\n```python\n{task.render(node.variant)}```\n"
                f"Visible test results (more tests exist that you cannot see):\n{checks}\n\n"
                f"You may change exactly ONE editable site to one of its options:\n{opts}\n\n"
                'Choose the single edit most likely to make the module fully correct. '
                'Reply with JSON only: {"site": <int>, "option": <int>}')

    def propose(self, node, rng: random.Random):
        import hashlib
        seed = rng.randrange(2**31)
        prompt = self.prompt(node)
        key = hashlib.sha256(f"{self.model}|nothink|{seed}|{prompt}".encode()).hexdigest()
        with self.lock:
            reply = self.cache.get(key)
        if reply is None:
            r = self.client.chat.completions.create(
                model=self.model, temperature=0.8, seed=seed, max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
                # Nemotron 3 is a reasoning model; with thinking on it exhausts the token budget
                # before answering. The executor gets one direct decision per call.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            reply = r.choices[0].message.content or ""
            with self.lock:
                self.cache[key] = reply
                self.stats["api_calls"] += 1
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "model": self.model, "seed": seed, "reply": reply}) + "\n")
        else:
            with self.lock:
                self.stats["cache_hits"] += 1
        task = TASKS[node.task]
        try:
            j = reply[reply.rfind("{"): reply.rfind("}") + 1]
            d = json.loads(j)
            site, opt = int(d["site"]), int(d.get("option", d.get("variant")))
            if not (0 <= site < len(task.sites) and 0 <= opt < len(task.sites[site])):
                raise ValueError("out of range")
            with self.lock:
                self.stats["valid"] += 1
            return (site, opt)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            with self.lock:
                self.stats["parse_fail"] += 1
            return (0, node.variant[0])  # no-op edit; the call is still charged
