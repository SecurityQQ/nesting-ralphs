"""TraceLoss: a small transformer that reads (Definition of Done, trace step embeddings, budget) and outputs
V = P(verified success within budget). loss_agent = 1 - V.

Embeddings: frozen BAAI/bge-small-en-v1.5. Training target: rollout label q. Loss: MSE (V - q)^2.
Evaluation: 5-fold cross-validation grouped by task (no task appears in train and test of the same fold).
Usage: PYTHONPATH=. uv run python -m trace_critic.traceloss
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .critic import Muon
from .nemo import dod_summary, step_text
from .tasks_v2 import FTASKS, TEST2, TRAIN2, VALID2

OUT = Path("runs/v2")
EMB_MODEL = "BAAI/bge-small-en-v1.5"
MAX_STEPS = 12
_emb = None


def embedder():
    global _emb
    if _emb is None:
        from sentence_transformers import SentenceTransformer
        _emb = SentenceTransformer(EMB_MODEL, device="cpu")
    return _emb


def embed(texts):
    return embedder().encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def dod_text(task, dod):
    return f"TASK: {task.goal}\nDEFINITION OF DONE: {dod_summary(dod)}"


class TraceLoss(nn.Module):
    def __init__(self, d_in=384, d=64, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.tok_type = nn.Embedding(2, d)  # 0 = DoD token, 1 = step token
        self.pos = nn.Embedding(MAX_STEPS + 2, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.budget = nn.Linear(1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, zdod, zsteps, mask, budget):
        b, t, _ = zsteps.shape
        x = torch.cat([self.proj(zdod)[:, None] + self.tok_type.weight[0], self.proj(zsteps) + self.tok_type.weight[1]], 1)
        x = x + self.pos(torch.arange(t + 1))[None]
        c = self.cls.expand(b, 1, -1) + self.budget(budget[:, None])[:, None]
        x = torch.cat([c, x], 1)
        pad = torch.cat([torch.zeros(b, 2, dtype=torch.bool), ~mask], 1)
        h = self.enc(x, src_key_padding_mask=pad)
        return torch.sigmoid(self.head(h[:, 0])).squeeze(-1)


def batchify(items):
    t = max(len(i["zs"]) for i in items)
    zs = np.zeros((len(items), t, 384), np.float32)
    mask = np.zeros((len(items), t), bool)
    for j, i in enumerate(items):
        zs[j, :len(i["zs"])] = i["zs"]
        mask[j, :len(i["zs"])] = True
    return (torch.tensor(np.stack([i["zdod"] for i in items])), torch.tensor(zs), torch.tensor(mask),
            torch.tensor([i["B"] / 4 for i in items], dtype=torch.float32))


def train_model(items, opt="adamw", epochs=80, seed=0, bs=32):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = TraceLoss()
    if opt == "muon":
        mats = [p for n, p in m.named_parameters() if p.ndim == 2 and "enc" in n]
        rest = [p for n, p in m.named_parameters() if not (p.ndim == 2 and "enc" in n)]
        opts = [Muon(mats, lr=0.02), torch.optim.AdamW(rest, lr=2e-3, weight_decay=1e-2)]
    else:
        opts = [torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)]
    y = torch.tensor([i["q"] for i in items], dtype=torch.float32)
    curve = []
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(len(items))
        tot = 0
        for k in range(0, len(idx), bs):
            j = idx[k:k + bs]
            loss = ((m(*batchify([items[i] for i in j])) - y[j]) ** 2).mean()
            for o in opts:
                o.zero_grad()
            loss.backward()
            for o in opts:
                o.step()
            tot += loss.item() * len(j)
        curve.append(tot / len(items))
    m.eval()
    return m, curve


def predict(m, items):
    with torch.no_grad():
        out = []
        for k in range(0, len(items), 64):
            out.append(m(*batchify(items[k:k + 64])).numpy())
        return np.concatenate(out) if out else np.zeros(0)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(384 * 2 + 1, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, zdod, zs, mask, budget):
        last = zs[torch.arange(len(zs)), mask.sum(1) - 1]
        return torch.sigmoid(self.net(torch.cat([zdod, last, budget[:, None]], 1))).squeeze(-1)


def train_mlp(items, epochs=80, seed=0):
    torch.manual_seed(seed)
    m = MLP()
    o = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    y = torch.tensor([i["q"] for i in items], dtype=torch.float32)
    X = batchify(items)
    for _ in range(epochs * 3):
        m.train()
        loss = ((m(*X) - y) ** 2).mean()
        o.zero_grad()
        loss.backward()
        o.step()
    m.eval()
    return m


def metrics(pred, items):
    q = np.array([i["q"] for i in items])
    pred = np.asarray(pred, float)
    out = {"n": len(items), "mse": float(((pred - q) ** 2).mean())}
    agree = tot = 0
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            if items[a]["task"] == items[b]["task"] and abs(q[a] - q[b]) > 0.3:
                tot += 1
                agree += (pred[a] - pred[b]) * (q[a] - q[b]) > 0
    out["within_task_rank_acc"] = agree / tot if tot else None
    out["within_task_pairs"] = tot
    pos, neg = pred[q > 0], pred[q == 0]
    out["auroc_any_success"] = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])) if len(pos) and len(neg) else None
    return out


def build_items():
    D = json.loads((OUT / "dod.json").read_text())
    S = [json.loads(l) for l in open(OUT / "states.jsonl")]
    dtexts = {n: dod_text(FTASKS[n], D[n]) for n in D}
    zd = dict(zip(dtexts, embed(dtexts.values())))
    texts = [step_text(st) for s in S for st in s["prefix"]["steps"][-MAX_STEPS:]]
    Z = embed(texts)
    items, k = [], 0
    for s in S:
        n = len(s["prefix"]["steps"][-MAX_STEPS:])
        last = s["prefix"]["steps"][-1]["dod"]
        items.append({"task": s["task"], "split": s["split"], "q": s["q"], "B": s["B"], "zdod": zd[s["task"]],
                      "zs": Z[k:k + n], "dod_frac": last["passed"] / max(1, last["total"]),
                      "verify_fails": s["prefix"]["verify_fails"]})
        k += n
    return items, zd


EPOCH_GRID = (5, 10, 20, 40)


def pick_epochs(train, opt, k=4):
    """Inner grouped CV on the training tasks only."""
    tasks = sorted({i["task"] for i in train})
    fold = {t: j % k for j, t in enumerate(tasks)}
    best = None
    for ep in EPOCH_GRID:
        se = n = 0
        for f in range(k):
            tr = [i for i in train if fold[i["task"]] != f]
            va = [i for i in train if fold[i["task"]] == f]
            if not tr or not va:
                continue
            m, _ = train_model(tr, opt, epochs=ep)
            se += float(((predict(m, va) - np.array([i["q"] for i in va])) ** 2).sum())
            n += len(va)
        if best is None or se / n < best[1]:
            best = (ep, se / n)
    return best[0]


def baselines(train, test):
    mean_q = float(np.mean([i["q"] for i in train]))
    return {
        "const_train_mean": [mean_q] * len(test),
        "dod_pass_fraction": [i["dod_frac"] for i in test],
        "cosine_last_step_to_DoD": [float((i["zs"][-1] @ i["zdod"] + 1) / 2) for i in test],
    }


def main():
    items, _ = build_items()
    tasks = sorted({i["task"] for i in items})
    folds = {t: k % 5 for k, t in enumerate(tasks)}
    preds = {k: np.zeros(len(items)) for k in ("traceloss_adamw", "traceloss_muon", "mlp_last_step", "const_train_mean",
                                               "dod_pass_fraction", "cosine_last_step_to_DoD")}
    curves, chosen = {}, {}
    fold_auroc = {k: [] for k in preds}
    for f in range(5):
        tr = [i for i in items if folds[i["task"]] != f]
        te_idx = [j for j, i in enumerate(items) if folds[i["task"]] == f]
        te = [items[j] for j in te_idx]
        if not te:
            continue
        for opt in ("adamw", "muon"):
            ep = pick_epochs(tr, opt)
            chosen.setdefault(opt, []).append(ep)
            m, c = train_model(tr, opt, epochs=ep)
            curves.setdefault(opt, []).append(c)
            preds[f"traceloss_{opt}"][te_idx] = predict(m, te)
        preds["mlp_last_step"][te_idx] = predict(train_mlp(tr), te)
        for k, v in baselines(tr, te).items():
            preds[k][te_idx] = v
        for k in preds:
            a = metrics(preds[k][te_idx], te)["auroc_any_success"]
            if a is not None:
                fold_auroc[k].append(a)
    cv = {k: {**metrics(v, items), "auroc_fold_mean": float(np.mean(fold_auroc[k])) if fold_auroc[k] else None}
          for k, v in preds.items()}
    # Final model for the controller: trained on train+valid task families only; test tasks sealed.
    trv = [i for i in items if i["split"] in ("train", "valid")]
    test = [i for i in items if i["split"] == "test"]
    final_ep = pick_epochs(trv, "adamw")
    final, fcurve = train_model(trv, "adamw", epochs=final_ep)
    torch.save(final.state_dict(), OUT / "traceloss.pt")
    sealed = {"traceloss_adamw": metrics(predict(final, test), test),
              "mlp_last_step": metrics(predict(train_mlp(trv), test), test)}
    for k, v in baselines(trv, test).items():
        sealed[k] = metrics(v, test)
    res = {"embedding_model": EMB_MODEL, "n_states": len(items), "folds": "5-fold grouped by task (20 tasks)",
           "cv": cv, "sealed_test_with_final_model": sealed, "epochs_chosen_cv": chosen, "final_epochs": final_ep,
           "split_sizes": {s: sum(i["split"] == s for i in items) for s in ("train", "valid", "test")}}
    (OUT / "traceloss_results.json").write_text(json.dumps(res, indent=1))
    (OUT / "traceloss_curves.json").write_text(json.dumps({"cv": curves, "final": fcurve}))
    for k, v in cv.items():
        print(f"{k:26s} mse={v['mse']:.4f} rank={v['within_task_rank_acc']} auroc_fold={v['auroc_fold_mean']}")
    print("epochs", chosen, "final", final_ep)
    print("sealed:", {k: round(v["mse"], 4) for k, v in sealed.items()})


if __name__ == "__main__":
    main()
