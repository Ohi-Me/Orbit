"""
Sequence models (PyTorch) -- the deep-learning tier of the model comparison.

WHY THESE EXIST
---------------
The platform's research question is whether added complexity earns its keep.
Asking that of logistic regression vs. XGBoost only covers tabular models
that see one date at a time. A sequence model sees the *path* -- how a
factor got to its current value, not just where it is -- which is the one
genuinely new hypothesis a deep model brings to this problem.

They are here to be beaten as easily as to win. On a few thousand rows with
a low signal-to-noise ratio, the honest prior is that they will NOT beat
XGBoost, and reporting that clearly is a real result rather than a failure.

DESIGN CHOICES FORCED BY SMALL DATA
  * Both models are deliberately tiny (~1-2 hidden layers, 32-64 units).
    A large model on 3,000 samples memorizes noise; the comparison would
    then measure regularization strength, not architecture.
  * Early stopping on a chronological tail of the training window -- never
    on the test fold, which would leak the very thing being measured.
  * Feature standardization uses training statistics only, applied to the
    validation and test slices unchanged.
  * Seeded end to end so a rerun reproduces the number in the report.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover - capability-probed upstream
    HAS_TORCH = False
    torch = None
    nn = object


SEQ_LEN = 20  # trading days of history per sample (~1 month)


if HAS_TORCH:

    class LSTMClassifier(nn.Module):
        """One-layer LSTM over the factor sequence, final hidden state -> logit.

        A single layer is deliberate: stacking layers on this sample size
        reliably overfits, and the second layer's capacity is not what the
        experiment is testing.
        """

        def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.3):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
            self.norm = nn.LayerNorm(hidden)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, (h, _) = self.lstm(x)
            h = self.norm(h[-1])
            return self.head(self.drop(h)).squeeze(-1)

    class TransformerClassifier(nn.Module):
        """Small Transformer encoder with mean pooling over time.

        Learned positional embeddings rather than sinusoidal: the sequence is
        only 20 steps and fixed-length, so there is nothing to extrapolate to
        and the learned version is simpler to reason about.
        """

        def __init__(
            self,
            n_features: int,
            d_model: int = 48,
            n_heads: int = 4,
            n_layers: int = 1,
            dropout: float = 0.3,
            seq_len: int = SEQ_LEN,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            h = self.input_proj(x) + self.pos[:, : x.size(1), :]
            h = self.encoder(h)
            h = self.norm(h.mean(dim=1))
            return self.head(self.drop(h)).squeeze(-1)


def build_sequences(
    df, feature_cols: list[str], seq_len: int = SEQ_LEN, label_col: str = "fwd_return_21d"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Turn a long panel into (X, y, returns, index_rows) sequence tensors.

    A sample for (ticker, date t) is the seq_len days of factor values ENDING
    at t, so every sequence is strictly historical with respect to its label.
    Sequences are built per ticker and never span a ticker boundary.

    Returns
        X            (n, seq_len, n_features)
        y            (n,)  binary direction label
        fwd_returns  (n,)  the raw forward return, for signal-return analysis
        index_rows   list of (date, ticker) aligning X back to the panel
    """
    Xs, ys, rets, idx = [], [], [], []
    for ticker, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date")
        feats = grp[feature_cols].to_numpy(dtype=np.float32)
        labels = grp[label_col].to_numpy(dtype=np.float32)
        dates = grp["date"].to_numpy()
        valid = ~np.isnan(feats).any(axis=1) & ~np.isnan(labels)
        for i in range(seq_len - 1, len(grp)):
            window = slice(i - seq_len + 1, i + 1)
            if not valid[window].all():
                continue
            Xs.append(feats[window])
            ys.append(1.0 if labels[i] > 0 else 0.0)
            rets.append(labels[i])
            idx.append((dates[i], ticker))
    if not Xs:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            [],
        )
    return (
        np.stack(Xs).astype(np.float32),
        np.array(ys, dtype=np.float32),
        np.array(rets, dtype=np.float32),
        idx,
    )


def train_sequence_model(
    kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int = 42,
    max_epochs: int = 60,
    patience: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """Train one sequence model and return (test_probabilities, train_info).

    The validation split is the chronological TAIL of the training window,
    not a random subset: a random split would put samples from after the
    validation point into training, which is the same leakage the walk-forward
    scheme exists to prevent, reintroduced one level down.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(False)

    n_features = X_train.shape[2]
    n_val = max(1, int(len(X_train) * val_fraction))
    if len(X_train) - n_val < 32:  # too small to hold out; train on everything
        n_val = 0

    # Standardize on training statistics only, computed across time and
    # samples jointly so each feature keeps its relative scale through time.
    flat = X_train.reshape(-1, n_features)
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd == 0] = 1.0

    def _norm(a):
        return ((a - mu) / sd).astype(np.float32)

    if n_val:
        X_fit, y_fit = _norm(X_train[:-n_val]), y_train[:-n_val]
        X_val, y_val = _norm(X_train[-n_val:]), y_train[-n_val:]
    else:
        X_fit, y_fit = _norm(X_train), y_train
        X_val, y_val = None, None

    model = (
        LSTMClassifier(n_features)
        if kind == "lstm"
        else TransformerClassifier(n_features, seq_len=X_train.shape[1])
    )

    # Class weighting: the direction label is rarely balanced, and an
    # unweighted model on a 55/45 split learns to predict the majority class
    # and reports it as accuracy.
    pos_rate = float(y_fit.mean()) if len(y_fit) else 0.5
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate if 0 < pos_rate < 1 else 1.0])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.from_numpy(X_fit)
    yt = torch.from_numpy(y_fit.astype(np.float32))
    Xv = torch.from_numpy(X_val) if X_val is not None else None
    yv = torch.from_numpy(y_val.astype(np.float32)) if y_val is not None else None

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    n = len(Xt)
    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(idx)
        train_loss = total / max(n, 1)

        if Xv is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(Xv), yv))
        else:
            val_loss = train_loss

        history.append({"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(_norm(X_test)))).numpy()

    return probs, {
        "architecture": kind,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "epochs_trained": len(history),
        "best_val_loss": round(float(best_val), 5),
        "early_stopped": epochs_without_improvement >= patience,
        "n_train_samples": int(len(X_fit)),
        "n_val_samples": int(n_val),
        "positive_class_rate": round(pos_rate, 4),
        "seed": seed,
        "note": "Validation split is the chronological tail of the training "
        "window; standardization uses training statistics only.",
    }
