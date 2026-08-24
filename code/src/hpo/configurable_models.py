"""
Hyperparameter-configurable AE and LSTM-AE detectors used by the HPO runners.

These are intentionally separate from src/models.py so the deployed paper
detector (AutoencoderDetector / LSTMAEDetector with their fixed CONFIG values)
remains exactly reproducible. The HPO variants accept arbitrary architecture,
optimizer, loss, and threshold strategies via a single config dict.

The two detectors share the F1-optimal-threshold and percentile-threshold
calibration logic with the main detectors via src.eval_utils.find_optimal_threshold.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..eval_utils import find_optimal_threshold, set_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════════════
# Loss / optimizer / activation registries
# ════════════════════════════════════════════════════════════════════
_LOSSES = {
    "mse":   nn.MSELoss(),
    "mae":   nn.L1Loss(),
    "huber": nn.SmoothL1Loss(),
}

_ACTIVATIONS = {
    "relu":       nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu":        nn.ELU,
    "tanh":       nn.Tanh,
    "selu":       nn.SELU,
}


def _make_optimizer(name: str, params, lr: float, wd: float):
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {name}")


# ════════════════════════════════════════════════════════════════════
# Configurable feed-forward AE
# ════════════════════════════════════════════════════════════════════
class _ConfigurableAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, dropout: float,
                 activation: str, use_batchnorm: bool):
        super().__init__()
        act_fn = _ACTIVATIONS[activation]

        enc, prev = [], input_dim
        for h in hidden_dims:
            enc.append(nn.Linear(prev, h))
            if use_batchnorm:
                enc.append(nn.BatchNorm1d(h))
            enc.append(act_fn())
            if dropout > 0:
                enc.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*enc)

        dec, prev = [], hidden_dims[-1]
        for h in list(hidden_dims)[-2::-1]:
            dec.append(nn.Linear(prev, h))
            if use_batchnorm:
                dec.append(nn.BatchNorm1d(h))
            dec.append(act_fn())
            if dropout > 0:
                dec.append(nn.Dropout(dropout))
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ConfigurableAEDetector:
    """One AE instance trained from a config dict; reuse across clean+poisoned."""

    def __init__(self, input_dim: int, config: dict):
        self.cfg = config
        self.input_dim = input_dim
        self.threshold: float | None = None
        self.train_time: float = 0.0
        self.stopped_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.model: _ConfigurableAE | None = None

    def _build(self):
        set_seed(self.cfg["seed"])
        return _ConfigurableAE(
            input_dim     = self.input_dim,
            hidden_dims   = list(self.cfg["hidden_dims"]),
            dropout       = float(self.cfg["dropout"]),
            activation    = str(self.cfg["activation"]),
            use_batchnorm = bool(self.cfg["use_batchnorm"]),
        ).to(DEVICE)

    def fit(self, X_train_normal: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        cfg = self.cfg
        set_seed(cfg["seed"])
        self.model = self._build()

        X_tr = torch.tensor(X_train_normal, dtype=torch.float32).to(DEVICE)
        opt = _make_optimizer(cfg["optimizer"], self.model.parameters(),
                              float(cfg["lr"]), float(cfg["weight_decay"]))
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                   patience=5, factor=0.5)
        crit = _LOSSES[cfg["loss_fn"]]

        # 90/10 internal split for early-stopping loss
        n = len(X_tr)
        vs = max(1, int(0.1 * n))
        X_val_internal = X_tr[:vs]
        X_train_split = X_tr[vs:]
        loader = DataLoader(TensorDataset(X_train_split),
                            batch_size=int(cfg["batch_size"]),
                            shuffle=True)

        best_vl, bad, best_state = float("inf"), 0, None
        t0 = time.time()
        last_epoch = 0
        for epoch in range(int(cfg["epochs"])):
            last_epoch = epoch + 1
            self.model.train()
            for (xb,) in loader:
                opt.zero_grad()
                rec = self.model(xb)
                loss = crit(rec, xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = crit(self.model(X_val_internal), X_val_internal).item()
            sch.step(vl)
            if vl < best_vl - 1e-5:
                best_vl, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(cfg["patience"]):
                    break
        self.train_time = time.time() - t0
        self.stopped_epoch = last_epoch
        self.best_val_loss = float(best_vl)
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Threshold calibration on the mixed validation fold
        val_errors = self._recon_error(X_val)
        self._calibrate_threshold(val_errors, y_val)

    def _recon_error(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        bs = int(self.cfg["batch_size"])
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.tensor(X[i:i + bs], dtype=torch.float32).to(DEVICE)
                rec = self.model(xb)
                err = ((rec - xb) ** 2).mean(dim=1).cpu().numpy()
                out.append(err)
        return np.concatenate(out) if out else np.empty(0)

    def _calibrate_threshold(self, val_errors: np.ndarray, y_val: np.ndarray):
        strat = self.cfg.get("threshold_strategy", "f1_optimal")
        if strat == "f1_optimal":
            self.threshold = find_optimal_threshold(y_val, val_errors)
        elif strat == "percentile_95":
            normal = val_errors[y_val == 0]
            self.threshold = float(np.percentile(normal, 95))
        elif strat == "percentile_99":
            normal = val_errors[y_val == 0]
            self.threshold = float(np.percentile(normal, 99))
        else:
            self.threshold = find_optimal_threshold(y_val, val_errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._recon_error(X)
        return (scores >= self.threshold).astype(int), scores


# ════════════════════════════════════════════════════════════════════
# Configurable LSTM-AE
# ════════════════════════════════════════════════════════════════════
class _ConfigurableLSTMAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, dropout: float):
        super().__init__()
        d = float(dropout) if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=d)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers,
                               batch_first=True, dropout=d)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        # Reverse-order reconstruction (Sutskever-style)
        dec_in = torch.flip(x, dims=[1])
        dec_out, _ = self.decoder(dec_in, (h, c))
        return torch.flip(self.out(dec_out), dims=[1])


class ConfigurableLSTMAEDetector:
    def __init__(self, input_dim: int, config: dict):
        self.cfg = config
        self.input_dim = input_dim
        self.window = int(config["window"])
        self.threshold: float | None = None
        self.train_time: float = 0.0
        self.stopped_epoch: int = 0
        self.best_val_loss: float = float("inf")
        self.model: _ConfigurableLSTMAE | None = None

    def _build(self):
        set_seed(self.cfg["seed"])
        return _ConfigurableLSTMAE(
            input_dim  = self.input_dim,
            hidden_dim = int(self.cfg["hidden_dim"]),
            num_layers = int(self.cfg["num_layers"]),
            dropout    = float(self.cfg["dropout"]),
        ).to(DEVICE)

    def _seq(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        W = self.window
        if len(X) < W:
            return np.empty((0, W, X.shape[1]), dtype=np.float32)
        return np.stack([X[i:i + W] for i in range(len(X) - W + 1)])

    def _seq_labels(self, y) -> np.ndarray:
        W = self.window
        y = np.asarray(y)
        if len(y) < W:
            return np.empty(0, dtype=int)
        return np.stack([1 if y[i:i + W].max() > 0 else 0
                         for i in range(len(y) - W + 1)])

    def fit(self, X_train_normal: np.ndarray, X_val_normal: np.ndarray,
            X_val_mixed: np.ndarray, y_val_mixed: np.ndarray):
        cfg = self.cfg
        set_seed(cfg["seed"])
        self.model = self._build()

        Xs_tr = torch.tensor(self._seq(X_train_normal), dtype=torch.float32)
        Xs_vn = torch.tensor(self._seq(X_val_normal),   dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xs_tr),
                            batch_size=int(cfg["batch_size"]),
                            shuffle=True)
        opt = _make_optimizer(cfg["optimizer"], self.model.parameters(),
                              float(cfg["lr"]), float(cfg["weight_decay"]))
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                   patience=5, factor=0.5)
        crit = _LOSSES[cfg["loss_fn"]]

        best_vl, bad, best_state = float("inf"), 0, None
        t0 = time.time()
        last_epoch = 0
        for epoch in range(int(cfg["epochs"])):
            last_epoch = epoch + 1
            self.model.train()
            for (xb,) in loader:
                xb = xb.to(DEVICE)
                opt.zero_grad()
                rec = self.model(xb)
                loss = crit(rec, xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            self.model.eval()
            with torch.no_grad():
                errs = []
                for i in range(0, len(Xs_vn), int(cfg["batch_size"])):
                    b = Xs_vn[i:i + int(cfg["batch_size"])].to(DEVICE)
                    errs.append(crit(self.model(b), b).item())
                vl = float(np.mean(errs)) if errs else float("inf")
            sch.step(vl)
            if vl < best_vl - 1e-5:
                best_vl, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= int(cfg["patience"]):
                    break
        self.train_time = time.time() - t0
        self.stopped_epoch = last_epoch
        self.best_val_loss = float(best_vl)
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Threshold on mixed validation (sequence-level labels)
        scores = self._scores(X_val_mixed)
        y_seq = self._seq_labels(y_val_mixed)
        n = min(len(scores), len(y_seq))
        self._calibrate_threshold(scores[:n], y_seq[:n])

    def _scores(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None
        self.model.eval()
        Xs = self._seq(X)
        if len(Xs) == 0:
            return np.empty(0)
        Xt = torch.tensor(Xs, dtype=torch.float32)
        bs = int(self.cfg["batch_size"])
        errs = []
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                b = Xt[i:i + bs].to(DEVICE)
                rec = self.model(b)
                errs.append(((rec - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(errs) if errs else np.empty(0)

    def _calibrate_threshold(self, scores: np.ndarray, y_seq: np.ndarray):
        strat = self.cfg.get("threshold_strategy", "f1_optimal")
        if strat == "f1_optimal":
            self.threshold = find_optimal_threshold(y_seq, scores)
        elif strat == "percentile_95":
            normal = scores[y_seq == 0] if (y_seq == 0).any() else scores
            self.threshold = float(np.percentile(normal, 95))
        elif strat == "percentile_99":
            normal = scores[y_seq == 0] if (y_seq == 0).any() else scores
            self.threshold = float(np.percentile(normal, 99))
        else:
            self.threshold = find_optimal_threshold(y_seq, scores)

    def predict(self, X: np.ndarray):
        scores = self._scores(X)
        return (scores >= self.threshold).astype(int), scores
