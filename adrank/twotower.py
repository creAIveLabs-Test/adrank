"""Stage 3: two-tower (dual encoder) retrieval.

User tower:  20-dim category-affinity vector (smoothed click counts from the
             TRAIN window only)  -> MLP -> 32-d embedding
Ad tower:    category one-hot + bid + smoothed historical CTR -> MLP -> 32-d

Training: clicked (user, ad) pairs as positives, IN-BATCH NEGATIVES with a
sampled-softmax loss — every other ad in the batch is a negative for this
user. This is the standard retrieval recipe (YouTube/Google two-tower papers).

Serving export is framework-free: both towers are tiny MLPs, so we dump the
weight matrices to models/twotower.npz and run the forward pass in numpy —
the server needs no torch (same spirit as exporting to ONNX, smaller hammer).

Offline eval: retrieval recall@K — of the test-window clicks, how often is
the clicked ad inside the user's top-K retrieved set? Compared against
random-K and popularity-K baselines.

Run:  uv run python -m adrank.twotower
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .db import get_session
from .eval import load_examples, time_split
from .models import CATEGORIES
from sqlmodel import select

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
EMB_DIM = 32
BATCH = 256
EPOCHS = 100
SMOOTH = 2.0


# ── Feature building (train window only — no future leakage) ─────────────────

def user_affinity(train, all_user_ids) -> dict[int, np.ndarray]:
    """20-dim smoothed per-category click-through profile per user."""
    imps = train.groupby(["user_id", "ad_category"]).size()
    clicks = train[train["clicked"] == 1].groupby(["user_id", "ad_category"]).size()
    out = {}
    for uid in all_user_ids:
        vec = np.zeros(len(CATEGORIES), dtype=np.float32)
        for ci, cat in enumerate(CATEGORIES):
            i = imps.get((uid, cat), 0)
            c = clicks.get((uid, cat), 0)
            vec[ci] = (c + SMOOTH * 0.03) / (i + SMOOTH)   # prior = global CTR
        out[uid] = vec
    return out


def ad_features(train, ads) -> dict[int, np.ndarray]:
    """category one-hot + bid + smoothed historical CTR per ad."""
    imps = train.groupby("ad_id").size()
    clicks = train[train["clicked"] == 1].groupby("ad_id").size()
    out = {}
    for ad in ads:
        onehot = np.zeros(len(CATEGORIES), dtype=np.float32)
        onehot[CATEGORIES.index(ad.category)] = 1.0
        i = imps.get(ad.id, 0)
        c = clicks.get(ad.id, 0)
        ctr = (c + SMOOTH * 0.03) / (i + SMOOTH)
        out[ad.id] = np.concatenate([onehot, [ad.bid / 5.0, ctr * 10.0]]).astype(np.float32)
    return out


# ── Model ─────────────────────────────────────────────────────────────────────

class Tower(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, EMB_DIM),
        )

    def forward(self, x):
        z = self.net(x)
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)   # L2-normalize


def train_two_tower(u_feats, a_feats, pairs, temp: float = 0.07):
    """pairs: list of (user_id, ad_id) clicked positives.

    logQ correction: in-batch negatives are sampled proportionally to click
    frequency, so popular ads are over-represented as negatives and the raw
    softmax punishes popularity. Subtracting log q(ad) from the logits makes
    the loss estimate the true softmax (Bengio & Senecal; used in the
    YouTube/Google two-tower retrieval papers).
    """
    user_tower = Tower(len(CATEGORIES))
    ad_tower = Tower(len(CATEGORIES) + 2)
    opt = torch.optim.Adam(
        list(user_tower.parameters()) + list(ad_tower.parameters()), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    pairs = np.array(pairs)
    counts = {}
    for _, a in pairs:
        counts[a] = counts.get(a, 0) + 1
    log_q = {a: np.log(c / len(pairs)) for a, c in counts.items()}
    for epoch in range(EPOCHS):
        np.random.shuffle(pairs)
        losses = []
        for i in range(0, len(pairs), BATCH):
            batch = pairs[i: i + BATCH]
            if len(batch) < 8:
                continue
            U = torch.tensor(np.stack([u_feats[u] for u, _ in batch]))
            A = torch.tensor(np.stack([a_feats[a] for _, a in batch]))
            logits = user_tower(U) @ ad_tower(A).T / temp   # (B, B)
            lq = torch.tensor([log_q[a] for _, a in batch], dtype=torch.float32)
            logits = logits - lq[None, :]                    # logQ correction
            labels = torch.arange(len(batch))                # diagonal = positive
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach()))
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}  in-batch softmax loss {np.mean(losses):.4f}")
    return user_tower, ad_tower


def export_npz(user_tower, ad_tower, path: Path) -> None:
    """Framework-free export: weight matrices for a numpy forward pass."""
    tensors = {}
    for prefix, tower in (("u", user_tower), ("a", ad_tower)):
        for j, layer in enumerate(l for l in tower.net if isinstance(l, nn.Linear)):
            tensors[f"{prefix}_W{j}"] = layer.weight.detach().numpy()
            tensors[f"{prefix}_b{j}"] = layer.bias.detach().numpy()
    np.savez(path, **tensors)


def numpy_forward(x: np.ndarray, npz, prefix: str) -> np.ndarray:
    h = np.maximum(x @ npz[f"{prefix}_W0"].T + npz[f"{prefix}_b0"], 0)
    z = h @ npz[f"{prefix}_W1"].T + npz[f"{prefix}_b1"]
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


# ── Eval: retrieval recall@K ──────────────────────────────────────────────────

def recall_at_k(test, train, u_embs, a_embs, ad_ids, k: int, rng) -> tuple[float, float, float]:
    """Of test clicks, how often is the clicked ad in the user's top-K?
    Returns (two-tower, popularity, random) recall. Popularity is ranked by
    TRAIN-window clicks — ranking by test impressions would leak the serving
    policy's own choices into the baseline."""
    train_clicks = train[train["clicked"] == 1].groupby("ad_id").size()
    pop_order = np.argsort(-train_clicks.reindex(ad_ids).fillna(0).to_numpy())
    pop_topk = set(np.array(ad_ids)[pop_order[:k]])
    ad_matrix = np.stack([a_embs[a] for a in ad_ids])

    pop_half = set(np.array(ad_ids)[pop_order[: k // 2]])
    hits_tt = hits_pop = hits_rand = hits_union = trials = 0
    for r in test[test["clicked"] == 1].itertuples():
        if r.user_id not in u_embs:
            continue
        trials += 1
        scores = ad_matrix @ u_embs[r.user_id]
        order = np.argsort(-scores)
        topk = set(np.array(ad_ids)[order[:k]])
        union = pop_half | set(np.array(ad_ids)[order[: k - len(pop_half)]])
        hits_tt += r.ad_id in topk
        hits_pop += r.ad_id in pop_topk
        hits_union += r.ad_id in union
        hits_rand += r.ad_id in set(rng.choice(ad_ids, size=k, replace=False))
    return (hits_tt / trials, hits_pop / trials, hits_union / trials,
            hits_rand / trials)


def main() -> None:
    from .models import Ad
    df = load_examples()
    train, test = time_split(df)
    with get_session() as s:
        ads = s.exec(select(Ad)).all()
    ad_ids = [a.id for a in ads]
    print(f"{len(df)} impressions · positives: train {train['clicked'].sum()} "
          f"/ test {test['clicked'].sum()}")

    u_feats = user_affinity(train, df["user_id"].unique())
    a_feats = ad_features(train, ads)
    pairs = [(r.user_id, r.ad_id) for r in train[train["clicked"] == 1].itertuples()]

    print("training two-tower (in-batch negatives)...")
    user_tower, ad_tower = train_two_tower(u_feats, a_feats, pairs)

    MODELS_DIR.mkdir(exist_ok=True)
    export_npz(user_tower, ad_tower, MODELS_DIR / "twotower.npz")
    npz = np.load(MODELS_DIR / "twotower.npz")
    u_embs = {u: numpy_forward(v[None, :], npz, "u")[0] for u, v in u_feats.items()}
    a_embs = {a: numpy_forward(v[None, :], npz, "a")[0] for a, v in a_feats.items()}
    # Persist serving-side inputs: user affinity vectors + ad embeddings
    train_clicks = train[train["clicked"] == 1].groupby("ad_id").size()
    np.savez(MODELS_DIR / "twotower_serving.npz",
             ad_ids=np.array(ad_ids),
             ad_embs=np.stack([a_embs[a] for a in ad_ids]),
             user_ids=np.array(list(u_feats.keys())),
             user_feats=np.stack(list(u_feats.values())),
             pop_counts=train_clicks.reindex(ad_ids).fillna(0).to_numpy())

    rng = np.random.default_rng(0)
    print(f"\n{'K':>4}{'two-tower':>12}{'popularity':>12}{'union':>10}{'random':>10}")
    for k in (25, 50, 100):
        tt, pop, union, rand = recall_at_k(test, train, u_embs, a_embs, ad_ids, k, rng)
        print(f"{k:>4}{tt:>12.3f}{pop:>12.3f}{union:>10.3f}{rand:>10.3f}")
    print(f"\nexports -> {MODELS_DIR}/twotower.npz, twotower_serving.npz")


if __name__ == "__main__":
    main()
