"""Value critic V_phi(z, B) ~ P(verified success within B | state, policy).

Small MLP. Ablation: AdamW everywhere vs Muon (Newton-Schulz orthogonalized momentum)
on the 2-D hidden weight matrix + AdamW on input/output layers and biases.
"""

import copy

import numpy as np
import torch
from torch import nn


def newton_schulz5(g, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (g.norm() + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    return x.T if transposed else x


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.setdefault("buf", torch.zeros_like(p.grad))
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                o = newton_schulz5(upd)
                p.add_(o, alpha=-g["lr"] * max(1.0, p.size(0) / p.size(1)) ** 0.5)


class Critic(nn.Module):
    def __init__(self, d_in, h=64):
        super().__init__()
        self.inp = nn.Linear(d_in, h)
        self.hid = nn.Linear(h, h)
        self.out = nn.Linear(h, 1)
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))

    def forward(self, x):
        x = (x - self.mu) / self.sd
        x = torch.relu(self.inp(x))
        x = torch.relu(self.hid(x))
        return torch.sigmoid(self.out(x)).squeeze(-1)


def train(Xtr, ytr, Xva, yva, opt="adamw", epochs=150, seed=0, bs=64):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = Critic(Xtr.shape[1])
    m.mu.copy_(torch.tensor(Xtr.mean(0)))
    m.sd.copy_(torch.tensor(Xtr.std(0) + 1e-3))
    if opt == "muon":
        opts = [Muon([m.hid.weight], lr=0.02),
                torch.optim.AdamW([m.inp.weight, m.inp.bias, m.hid.bias, m.out.weight, m.out.bias],
                                  lr=3e-3, weight_decay=1e-4)]
    else:
        opts = [torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=1e-4)]
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva)
    best, best_state, curve = 1e9, None, []
    for ep in range(epochs):
        m.train()
        idx = rng.permutation(len(Xt))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            loss = ((m(Xt[j]) - yt[j]) ** 2).mean()  # L_value = mean (V - q)^2
            for o in opts:
                o.zero_grad()
            loss.backward()
            for o in opts:
                o.step()
        m.eval()
        with torch.no_grad():
            tr = ((m(Xt) - yt) ** 2).mean().item()
            va = ((m(Xv) - yv) ** 2).mean().item()
        curve.append((ep, tr, va))
        if va < best:  # epoch selection on the VALID task family only
            best, best_state = va, copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    return m, curve


def predict(m, X):
    with torch.no_grad():
        return m(torch.tensor(X, dtype=torch.float32)).numpy()
