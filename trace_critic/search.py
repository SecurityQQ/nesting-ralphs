"""State graph, features, rollouts and the three controllers.

Budget = executor calls (one call proposes one action). Verifier calls are counted
separately and never consume executor budget. Every executed branch is charged,
including discarded ones.
"""

import hashlib
import itertools
import random
from dataclasses import dataclass, field, replace

import numpy as np

from .env import verify
from .tasks import TASKS

_ids = itertools.count()


@dataclass
class Node:
    task: str
    variant: tuple
    observed: dict  # visible checks only (what the executor sees)
    artifact: str
    parent: "Node|None" = None
    last_action: tuple | None = None
    history: tuple = ()  # actions taken so far
    verify_fails: int = 0
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def visible_pass(self):
        return all(self.observed.values())


def make_node(task_name, variant, parent=None, action=None):
    task = TASKS[task_name]
    h, res = verify(task_name, variant)  # verifier runs the checks; we expose only visible ones
    obs = {c: res[c] for c in task.visible}
    return Node(task_name, variant, obs, h, parent, action,
                (parent.history + (action,)) if parent else (),
                parent.verify_fails if parent else 0)


def apply(node, action):
    site, v = action
    var = list(node.variant)
    var[site] = v
    return make_node(node.task, tuple(var), node, action)


def verified(node):
    _, res = verify(node.task, node.variant)
    return all(res.values())


# ---------------------------------------------------------------- features
# Environment-only features (no narration, no task id, no hidden outcomes).
HASH_DIM = 32


def _hash_tokens(text):
    v = np.zeros(HASH_DIM, dtype=np.float32)
    for tok in text.replace("(", " ").replace(")", " ").replace(",", " ").split():
        v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % HASH_DIM] += 1.0
    return np.log1p(v)


def features(node, budget_left, b_ref=16):
    task = TASKS[node.task]
    obs = list(node.observed.values())
    hist = node.history
    last3 = hist[-3:]
    edited = {a[0] for a in hist}
    seen, n = set(), node
    states = []
    while n is not None:
        states.append(n.artifact)
        n = n.parent
    vis_frac = sum(obs) / len(obs)
    parent_frac = (sum(node.parent.observed.values()) / len(obs)) if node.parent else vis_frac
    edited_text = " ".join(task.sites[s][node.variant[s]] for s in sorted(edited))
    struct = np.array([
        vis_frac, float(all(obs)), len(obs) - sum(obs), vis_frac - parent_frac,
        len(hist) / b_ref, budget_left / b_ref, len(set(last3)) / max(1, len(last3)),
        len(set(states)) / len(states), node.verify_fails, len(edited) / len(task.sites),
        sum(1 for v in node.variant if v == 0) / len(task.sites), float(budget_left <= 0),
    ], dtype=np.float32)
    return np.concatenate([struct, _hash_tokens(edited_text)])


N_STRUCT = 12
FEAT_DIM = N_STRUCT + HASH_DIM


# ---------------------------------------------------------------- Ralph rollout (label source + baseline)
def ralph(node, executor, budget, rng, log=None):
    """Bounded Ralph: repeat the same loop; verify when the executor believes it's done.
    Returns (success, executor_calls, verify_calls)."""
    node = replace(node)  # never mutate the caller's checkpoint
    calls = vcalls = 0
    if node.visible_pass:
        vcalls += 1
        if verified(node):
            return True, calls, vcalls, node
        node.verify_fails += 1
    while calls < budget:
        a = executor.propose(node, rng)
        calls += 1
        child = apply(node, a)
        if log is not None:
            log.append({"node": child, "parent": node, "decision": "CONTINUE"})
        node = child
        if node.visible_pass:
            vcalls += 1
            if verified(node):
                if log is not None:
                    log.append({"node": node, "parent": node.parent, "decision": "VERIFIED"})
                return True, calls, vcalls, node
            node.verify_fails += 1
            if log is not None:
                log.append({"node": node, "parent": node.parent, "decision": "FLAG_GAP"})
    return False, calls, vcalls, node


# ---------------------------------------------------------------- graph search
def graph_search(root, executor, budget, rng, scorer, k=2, frontier_cap=8, gap_v=0.8):
    """Budgeted best-first search. scorer(node, budget_left) -> float.
    Returns dict with success, costs, and an event log for TRACE."""
    calls = vcalls = loops = gaps = 0
    events = []
    seen = {root.artifact}
    frontier = [root]
    root.score = scorer(root, budget)
    events.append(dict(id=root.id, parent=None, decision="ROOT", score=root.score, calls=0,
                       vis=sum(root.observed.values()) / len(root.observed), verified=None, action=None))
    current = root
    while calls + k <= budget and frontier:
        best = max(frontier, key=lambda n: (n.score, n.id))
        decision = "CONTINUE" if best is current or best.parent is current else "BACKTRACK"
        # Policy is stochastic: a checkpoint may be re-expanded; each expansion lowers its priority.
        best.expansions = getattr(best, "expansions", 0) + 1
        best.score -= EXPAND_PENALTY
        current = best
        children = []
        for _ in range(k):
            a = executor.propose(best, rng)
            calls += 1
            child = apply(best, a)
            if child.artifact in seen:  # exact repeat, deterministic checks => no new observation
                loops += 1
                events.append(dict(id=child.id, parent=best.id, decision="REPLAN(loop)", score=None,
                                   calls=calls, vis=None, verified=None, action=a))
                continue
            seen.add(child.artifact)
            children.append(child)
        for child in children:
            child.score = scorer(child, budget - calls)
            ver = None
            if child.visible_pass:
                vcalls += 1
                ver = verified(child)
                if ver:
                    events.append(dict(id=child.id, parent=best.id, decision="VERIFIED", score=child.score,
                                       calls=calls, vis=1.0, verified=True, action=child.last_action, via=decision))
                    return dict(success=True, calls=calls, vcalls=vcalls, loops=loops, gaps=gaps, events=events)
                child.verify_fails += 1
                child.score = scorer(child, budget - calls)
                if child.score >= gap_v:
                    gaps += 1
            events.append(dict(id=child.id, parent=best.id, decision=("FLAG_GAP" if ver is False else decision),
                               score=child.score, calls=calls, vis=sum(child.observed.values()) / len(child.observed),
                               verified=ver, action=child.last_action))
            frontier.append(child)
        frontier.sort(key=lambda n: (n.score, n.id), reverse=True)
        del frontier[frontier_cap:]
    events.append(dict(id=None, parent=None, decision="STOP_UNRESOLVED", score=None, calls=calls,
                       vis=None, verified=False, action=None))
    return dict(success=False, calls=calls, vcalls=vcalls, loops=loops, gaps=gaps, events=events)


EXPAND_PENALTY = 0.05


def rule_scorer(node, budget_left):
    """Fixed rules: visible pass fraction, penalize repeated actions and failed verifications."""
    obs = list(node.observed.values())
    rep = len(node.history[-3:]) - len(set(node.history[-3:]))
    return sum(obs) / len(obs) - 0.1 * rep - 0.2 * node.verify_fails
