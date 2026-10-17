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
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.environ["CRUSOE_INFERENCE_BASE_URL"],
                             api_key=os.environ["CRUSOE_INFERENCE_API_KEY"])
        self.model = os.environ["CRUSOE_INFERENCE_MODEL"]
        self.policy_id = f"nemotron:{self.model}"

    def propose(self, node, rng: random.Random):
        task = TASKS[node.task]
        opts = "\n".join(f"site {i}: " + " | ".join(f"[{j}] {v!r}" for j, v in enumerate(s))
                         for i, s in enumerate(task.sites))
        prompt = (f"Goal: {task.goal}\nCurrent source:\n{task.render(node.variant)}\n"
                  f"Observed checks: {json.dumps(node.observed)}\nEditable sites and options:\n{opts}\n"
                  'Reply with JSON only: {"site": <int>, "variant": <int>}')
        r = self.client.chat.completions.create(model=self.model, temperature=0.8,
                                                seed=rng.randrange(2**31),
                                                messages=[{"role": "user", "content": prompt}])
        txt = r.choices[0].message.content
        d = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        return (int(d["site"]), int(d["variant"]))
