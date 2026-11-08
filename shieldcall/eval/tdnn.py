"""Small TDNN (x-vector style) spoof/scam-voice classifier with optional training tricks. CPU friendly."""
from __future__ import annotations
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam; return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class TDNN(nn.Module):
    def __init__(self, d_in, h=128, emb=64, drop=0.2, n_dom=0):
        super().__init__()
        self.dom = nn.Sequential(nn.Linear(emb, 64), nn.ReLU(), nn.Linear(64, n_dom)) if n_dom else None
        self.c1 = nn.Conv1d(d_in, h, 5, padding=2); self.b1 = nn.BatchNorm1d(h)
        self.c2 = nn.Conv1d(h, h, 3, padding=2, dilation=2); self.b2 = nn.BatchNorm1d(h)
        self.c3 = nn.Conv1d(h, h, 3, padding=3, dilation=3); self.b3 = nn.BatchNorm1d(h)
        self.fc = nn.Linear(2 * h, emb); self.drop = nn.Dropout(drop); self.out = nn.Linear(emb, 1)

    def embed(self, x):                      # x: (B, D, T)
        x = Fn.relu(self.b1(self.c1(x))); x = Fn.relu(self.b2(self.c2(x))); x = Fn.relu(self.b3(self.c3(x)))
        s = torch.cat([x.mean(2), x.std(2) + 1e-5], 1)
        return Fn.relu(self.fc(self.drop(s)))

    def forward(self, x, lam=0.0):
        e = self.embed(x)
        d = self.dom(_GRL.apply(e, lam)) if self.dom is not None else None
        return self.out(self.drop(e)).squeeze(1), (e if d is None else d)


def crop(f: np.ndarray, T: int, rng) -> np.ndarray:
    if len(f) < T:
        f = np.pad(f, ((0, T - len(f)), (0, 0)), mode="wrap")
    s = rng.randint(0, len(f) - T + 1)
    return f[s:s + T]


def specaug(x: torch.Tensor, rng, fm=8, tm=20):
    x = x.clone()
    for i in range(x.shape[0]):
        f0 = rng.randint(0, max(1, x.shape[1] - fm)); x[i, f0:f0 + rng.randint(0, fm + 1), :] = 0
        t0 = rng.randint(0, max(1, x.shape[2] - tm)); x[i, :, t0:t0 + rng.randint(0, tm + 1)] = 0
    return x


@torch.no_grad()
def score(model, F: np.ndarray) -> float:
    model.eval()
    x = torch.from_numpy(F.T[None].copy())
    return float(model(x)[0])
