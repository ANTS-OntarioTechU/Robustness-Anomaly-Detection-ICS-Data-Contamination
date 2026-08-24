"""
Anomaly detector zoo — 10 PyOD wrappers + two PyTorch detectors (Autoencoder, LSTM-AE).

Logic mirrors notebook cells 3.2 / 3.3 / 3.4 / 3.5 exactly. The only non-default
hyperparameters are `TUNED_PARAMS` (Phase 1 sensitivity study winners).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA as SkPCA

from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.pca import PCA as PCA_AD
from pyod.models.mcd import MCD
from pyod.models.abod import ABOD
from pyod.models.sod import SOD

from .config import CONFIG, TUNED_PARAMS
from .eval_utils import find_optimal_threshold, set_seed



# ─────── pyod 2.0.3 + sklearn 1.6+ compatibility shim ───────
# sklearn 1.6 introduced __sklearn_tags__ as a required method on estimators.
# pyod 2.0.3 hasn't been updated for this, so check_is_fitted inside pyod's
# decision_function raises AttributeError. Attach BaseEstimator's
# __sklearn_tags__ to each pyod class that's missing it.
from sklearn.base import BaseEstimator as _SkBE
if hasattr(_SkBE, "__sklearn_tags__"):
    for _cls in (IForest, OCSVM, LOF, CBLOF, KNN, HBOS, PCA_AD, MCD, ABOD, SOD):
        if not hasattr(_cls, "__sklearn_tags__"):
            _cls.__sklearn_tags__ = _SkBE.__sklearn_tags__

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PYOD_MODELS  = {"iforest", "svm", "lof", "cluster", "knn", "histogram",
                "pca", "mcd", "abod", "sod"}
TORCH_MODELS = {"autoencoder", "lstm_ae"}


# ────────────────────────── PyOD wrapper ────────────────────────────
class AnomalyDetector:
    """PyOD detector wrapped with F1-optimal threshold calibration."""

    def __init__(self, model_name: str, pyod_model=None,
                 contamination: float = CONFIG["CONTAMINATION"], seed: int = 42):
        self.model_name = model_name
        self.contamination = contamination
        self.seed = seed
        self.threshold: float | None = None
        self.model = pyod_model if pyod_model is not None else self._build_default()

    def _build_default(self):
        c = self.contamination
        tp = TUNED_PARAMS.get(self.model_name, {})
        m = self.model_name
        if   m == "iforest":   return IForest(contamination=c, random_state=self.seed)
        elif m == "svm":       return OCSVM(contamination=c,
                                             nu=tp.get("nu", 0.05),
                                             gamma=tp.get("gamma", "scale"))
        elif m == "lof":       return LOF(contamination=c, novelty=True)
        elif m == "cluster":   return CBLOF(contamination=c, random_state=self.seed)
        elif m == "knn":       return KNN(contamination=c)
        elif m == "histogram": return HBOS(contamination=c)
        elif m == "pca":
            ncomp = tp.get("n_components")
            if isinstance(ncomp, int):
                return PCA_AD(contamination=c, n_components=ncomp, random_state=self.seed)
            # placeholder; float variance ratio is resolved in train()
            return PCA_AD(contamination=c, random_state=self.seed)
        elif m == "mcd":       return MCD(contamination=c, random_state=self.seed)
        elif m == "abod":      return ABOD(contamination=c, method="fast", n_neighbors=10)
        elif m == "sod":       return SOD(contamination=c, n_neighbors=20, ref_set=10)
        raise ValueError(f"Unknown model: {m}")

    def train(self, X_train_normal, _unused, X_val, y_val):
        # Resolve PCA variance-ratio → integer component count at fit time
        if self.model_name == "pca":
            ncomp = TUNED_PARAMS.get("pca", {}).get("n_components")
            if isinstance(ncomp, float):
                tmp = SkPCA(n_components=ncomp, random_state=self.seed)
                tmp.fit(X_train_normal[:min(5000, len(X_train_normal))])
                self.model = PCA_AD(contamination=self.contamination,
                                    n_components=int(tmp.n_components_),
                                    random_state=self.seed)

        self.model.fit(X_train_normal)
        scores_val = self.model.decision_function(X_val)
        self.threshold = find_optimal_threshold(y_val, scores_val)

    def predict(self, X):
        return (self.model.decision_function(X) >= self.threshold).astype(int)

    def decision_scores(self, X):
        return self.model.decision_function(X)


# ─────────────────────── Feed-forward Autoencoder ───────────────────
_AE_ACT = {
    "relu":       nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu":        nn.ELU,
    "selu":       nn.SELU,
    "tanh":       nn.Tanh,
}


class _AEModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(256, 128, 64),
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        act_cls = _AE_ACT[activation]
        enc, prev = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), act_cls(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc)
        dec, prev = [], hidden_dims[-1]
        for h in list(hidden_dims)[-2::-1]:
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), act_cls(), nn.Dropout(dropout)]
            prev = h
        dec += [nn.Linear(prev, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AutoencoderDetector:
    def __init__(self, input_dim: int, hidden_dims=None, seed: int = 42):
        set_seed(seed)
        self.model = _AEModel(
            input_dim,
            hidden_dims=tuple(hidden_dims or CONFIG["AE_HIDDEN_DIMS"]),
            dropout=CONFIG["AE_DROPOUT"],
            activation=CONFIG.get("AE_ACTIVATION", "relu"),
        ).to(DEVICE)
        self.threshold: float | None = None
        self.seed = seed

    def train(self, X_train_normal, _unused, X_val, y_val):
        set_seed(self.seed)
        X_tr = torch.tensor(X_train_normal, dtype=torch.float32).to(DEVICE)

        opt  = optim.Adam(self.model.parameters(), lr=CONFIG["AE_LR"], weight_decay=1e-5)
        sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        crit = nn.MSELoss()

        # 90/10 split for early stopping
        n  = len(X_tr)
        vs = max(1, int(0.1 * n))
        X_val_normal = X_tr[:vs]
        X_tr2 = X_tr[vs:]
        loader = DataLoader(
            TensorDataset(X_tr2),
            batch_size=CONFIG.get("AE_BATCH_SIZE", 1024),
            shuffle=True,
        )

        best_vl, bad, best_state = float("inf"), 0, None
        for _ in range(CONFIG["AE_EPOCHS"]):
            self.model.train()
            for (xb,) in loader:
                opt.zero_grad()
                out = self.model(xb)
                loss = crit(out, xb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = crit(self.model(X_val_normal), X_val_normal).item()
            sch.step(vl)
            if vl < best_vl - 1e-5:
                best_vl, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= CONFIG["AE_PATIENCE"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Threshold on mixed validation
        self.model.eval()
        with torch.no_grad():
            Xv  = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
            rec = self.model(Xv)
            err = ((rec - Xv) ** 2).mean(dim=1).cpu().numpy()
        self.threshold = find_optimal_threshold(y_val, err)

    def decision_scores(self, X):
        self.model.eval()
        with torch.no_grad():
            Xt  = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            rec = self.model(Xt)
            err = ((rec - Xt) ** 2).mean(dim=1).cpu().numpy()
        return err

    def predict(self, X):
        return (self.decision_scores(X) >= self.threshold).astype(int)


# ─────────────────────────── LSTM-AE ────────────────────────────────
class _LSTMAEModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        d = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=d)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=d)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        dec_in = torch.flip(x, dims=[1])
        dec_out, _ = self.decoder(dec_in, (h, c))
        return torch.flip(self.out(dec_out), dims=[1])


class LSTMAEDetector:
    def __init__(self, input_dim: int, seed: int = 42):
        set_seed(seed)
        self.model = _LSTMAEModel(
            input_dim,
            hidden_dim=CONFIG["LSTM_AE_HIDDEN"],
            dropout=CONFIG["LSTM_AE_DROPOUT"],
        ).to(DEVICE)
        self.window = CONFIG["LSTM_AE_WINDOW"]
        self.threshold: float | None = None
        self.seed = seed
        self.input_dim = input_dim

    def _seq(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        W = self.window
        if len(X) < W:
            return np.empty((0, W, X.shape[1]), dtype=np.float32)
        return np.stack([X[i:i + W] for i in range(len(X) - W + 1)])

    def _create_sequence_labels(self, y) -> np.ndarray:
        W = self.window
        y = np.asarray(y)
        if len(y) < W:
            return np.empty(0, dtype=int)
        return np.stack([1 if y[i:i + W].max() > 0 else 0
                         for i in range(len(y) - W + 1)])

    def train(self, X_train_normal, X_val_normal, X_val, y_val):
        set_seed(self.seed)
        Xs_tr = torch.tensor(self._seq(X_train_normal), dtype=torch.float32)
        Xs_vn = torch.tensor(self._seq(X_val_normal),   dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(Xs_tr),
            batch_size=CONFIG.get("LSTM_AE_BATCH", 512),
            shuffle=True,
        )

        opt_name = CONFIG.get("LSTM_AE_OPTIMIZER", "adam").lower()
        _OPT = {"adam": optim.Adam, "adamw": optim.AdamW}[opt_name]
        opt  = _OPT(self.model.parameters(), lr=CONFIG["LSTM_AE_LR"], weight_decay=1e-5)
        sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        crit = nn.MSELoss()

        best_vl, bad, best_state = float("inf"), 0, None
        for _ in range(CONFIG["LSTM_AE_EPOCHS"]):
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
                for i in range(0, len(Xs_vn), 512):
                    b = Xs_vn[i:i + 512].to(DEVICE)
                    errs.append(crit(self.model(b), b).item())
                vl = float(np.mean(errs)) if errs else float("inf")
            sch.step(vl)
            if vl < best_vl - 1e-5:
                best_vl, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= CONFIG["LSTM_AE_PATIENCE"]:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Threshold on mixed validation (sequence-level labels)
        scores = self.decision_scores(X_val)
        y_seq  = self._create_sequence_labels(y_val)
        n = min(len(scores), len(y_seq))
        self.threshold = find_optimal_threshold(y_seq[:n], scores[:n])

    def decision_scores(self, X):
        self.model.eval()
        Xs = self._seq(X)
        if len(Xs) == 0:
            return np.empty(0)
        Xt = torch.tensor(Xs, dtype=torch.float32)
        errs = []
        with torch.no_grad():
            for i in range(0, len(Xt), 512):
                b = Xt[i:i + 512].to(DEVICE)
                rec = self.model(b)
                e = ((rec - b) ** 2).mean(dim=(1, 2)).cpu().numpy()
                errs.append(e)
        return np.concatenate(errs) if errs else np.empty(0)

    def predict(self, X):
        return (self.decision_scores(X) >= self.threshold).astype(int)


# ────────────────────────── factory ─────────────────────────────────
def create_model(name: str, input_dim: int, seed: int = 42):
    if name in PYOD_MODELS:
        return AnomalyDetector(name, contamination=CONFIG["CONTAMINATION"], seed=seed)
    if name == "autoencoder":
        return AutoencoderDetector(input_dim, seed=seed)
    if name == "lstm_ae":
        return LSTMAEDetector(input_dim, seed=seed)
    raise ValueError(f"Unknown model: {name}")
