# EE 460 Final Project – Heart Disease Predictor with MLP

import os, random, joblib, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.base")
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

# ---------------------------------------------------------------------
# 1. Load *balanced* & pre‑processed data
# ---------------------------------------------------------------------
DATA_NPZ = "/Users/georgeyang/Documents/USC/Spring 2025/EE 460/FinalProject/heart_balanced_preproc.npz"
assert os.path.exists(DATA_NPZ), f"Balanced dataset not found: {DATA_NPZ}"
loaded = np.load(DATA_NPZ)
X = loaded["X"].astype(np.float32)   # shape (N, F)
y = loaded["y"].astype(np.int64)     # shape (N,)
print(f"Loaded balanced data → X:{X.shape},  y:{y.shape},  classes:{np.unique(y)}")

# ---------------------------------------------------------------------
# 1b. Load *unbalanced* original data (for comparison)
# ---------------------------------------------------------------------
import pandas as pd

ORIG_CSV = "/Users/georgeyang/Documents/USC/Spring 2025/EE 460/FinalProject/heart_disease_uci.csv"

df_orig = pd.read_csv(ORIG_CSV, na_values="?")

df_orig.rename(columns={"num": "target"}, inplace=True)

df_orig.dropna(inplace=True)

bool_cols = df_orig.select_dtypes("bool").columns
df_orig[bool_cols] = df_orig[bool_cols].astype(int)

cat_cols = df_orig.select_dtypes(include=["object"]).columns.difference(["target"])
for col in cat_cols:
    df_orig[col] = pd.Categorical(df_orig[col]).codes

df_orig = df_orig.apply(pd.to_numeric, errors="coerce")
df_orig.dropna(inplace=True)

y_orig = df_orig["target"].astype(np.int64).values
X_orig = df_orig.drop(columns=["target"]).astype(np.float32).values
print(f"Loaded original data → X:{X_orig.shape},  y:{y_orig.shape},  class counts:{np.bincount(y_orig)}")

Xo_train, Xo_temp, yo_train, yo_temp = train_test_split(
    X_orig, y_orig, test_size=0.20, stratify=y_orig, random_state=SEED)
Xo_val, Xo_test, yo_val, yo_test = train_test_split(
    Xo_temp, yo_temp, test_size=0.50, stratify=yo_temp, random_state=SEED)



# ---------------------------------------------------------------------
# 2. Train / Val / Test split (80 / 10 / 10)
# ---------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)
print("Split sizes →", {k:v.shape[0] for k,v in {
    'train':y_train, 'val':y_val, 'test':y_test}.items()})

input_dim = X.shape[1]
num_class = len(np.unique(y))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 3. DataLoader helper
# ---------------------------------------------------------------------

def make_loader(X_arr, y_arr, batch_size=128, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X_arr), torch.from_numpy(y_arr))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# ---------------------------------------------------------------------
# 4. Optuna hyper‑parameter search
# ---------------------------------------------------------------------

class Net(nn.Module):
    """Dynamically sized 3‑hidden‑layer MLP."""
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg["h1"]), nn.ReLU(), nn.Dropout(cfg["drop"]),
            nn.Linear(cfg["h1"], cfg["h2"]), nn.ReLU(), nn.Dropout(cfg["drop"]),
            nn.Linear(cfg["h2"], cfg["h3"]), nn.ReLU(), nn.Dropout(cfg["drop"]),
            nn.Linear(cfg["h3"], num_class))
    def forward(self, x):
        return self.net(x)


def objective(trial):
    cfg = {
        "h1"  : trial.suggest_int("h1", 256, 512, step=64),
        "h2"  : trial.suggest_int("h2", 128, 256, step=32),
        "h3"  : trial.suggest_int("h3",  64, 128, step=16),
        "drop": trial.suggest_float("drop", 0.1, 0.3, step=0.05),
        "lr"  : trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "batch": trial.suggest_categorical("batch", [64, 128, 256]),
        "wd"  : trial.suggest_float("wd", 1e-6, 1e-3, log=True),
        "epochs": trial.suggest_int("epochs", 50, 150, step=10),
    }

    tr_loader = make_loader(X_train, y_train, batch_size=cfg["batch"])
    va_loader = make_loader(X_val,   y_val,   batch_size=cfg["batch"])

    model = Net(cfg).to(DEVICE)
    freq = np.bincount(y_train) + 1e-6
    cls_w = torch.tensor(np.log(freq.sum()/freq), dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.15)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    best_f1 = 0.0
    for ep in range(cfg["epochs"]):
        # train one epoch
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward(); optimizer.step()

        # validate
        model.eval(); preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                preds.extend(torch.argmax(model(xb.to(DEVICE)),1).cpu().numpy())
                trues.extend(yb.numpy())
        val_f1 = f1_score(trues, preds, average="weighted", zero_division=0)
        best_f1 = max(best_f1, val_f1)
        trial.report(val_f1, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return best_f1

print("\n=== Optuna search (balanced data) ===")
study = optuna.create_study(direction="maximize", study_name="heart_mlp_balanced")
study.optimize(objective, n_trials=40, show_progress_bar=True)

BEST = study.best_params
print("Best F1 :", study.best_value) 
print("Best params:", BEST)

# ---------------------------------------------------------------------
# 5. Define final network & utils
# ---------------------------------------------------------------------

class FinalNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg["h1"]), nn.ReLU(), nn.Dropout(cfg["drop"]),
            nn.Linear(cfg["h1"], cfg["h2"]), nn.ReLU(), nn.Dropout(cfg["drop"]),
            nn.Linear(cfg["h2"], cfg["h3"]), nn.ReLU(), nn.Dropout(cfg["drop"]),
            nn.Linear(cfg["h3"], num_class))
    def forward(self, x):
        return self.net(x)


def loader_pair(batch):
    return (make_loader(X_train, y_train, batch_size=batch, shuffle=True),
            make_loader(X_val,   y_val,   batch_size=batch, shuffle=False))

# ---------------------------------------------------------------------
# Utility: generate accuracy-vs-samples curve for任意数据集
# ---------------------------------------------------------------------
def acc_vs_samples(X_tr_full, y_tr_full, X_val_local, y_val_local, cfg, n_points=10, epochs=15):
    sample_fracs = np.linspace(0.1, 1.0, n_points)     # 10% → 100%
    acc_values   = []
    sizes        = []
    for frac in sample_fracs:
        n = int(len(y_tr_full) * frac)
        idx = np.random.choice(len(y_tr_full), n, replace=False)
        X_sub, y_sub = X_tr_full[idx], y_tr_full[idx]
        sizes.append(n)

        tr_loader = make_loader(X_sub, y_sub, batch_size=cfg["batch"])
        va_loader = make_loader(X_val_local, y_val_local, batch_size=cfg["batch"], shuffle=False)

        net = FinalNet(cfg).to(DEVICE)
        freq = np.bincount(y_sub) + 1e-6
        w    = torch.tensor(np.log(freq.sum()/freq), dtype=torch.float32, device=DEVICE)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.15)
        opt  = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=1e-4)

        for _ in range(epochs):
            net.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad(); loss = crit(net(xb), yb); loss.backward(); opt.step()

        acc = eval_fold(net, va_loader)
        acc_values.append(acc)

        del net; torch.cuda.empty_cache(); gc.collect()
    return sizes, acc_values


# ---------------------------------------------------------------------
# 6. Train with early stopping
# ---------------------------------------------------------------------

patience = 10
model   = FinalNet(BEST).to(DEVICE)
train_loader, val_loader = loader_pair(BEST["batch"])

freq = np.bincount(y_train) + 1e-6
cls_w = torch.tensor(np.log(freq.sum()/freq), dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.15)
optimizer = torch.optim.Adam(model.parameters(), lr=BEST["lr"], weight_decay=1e-4)

best_val_loss = float("inf"); wait = 0
train_loss_hist = []; val_loss_hist = []
train_f1_hist   = []; val_f1_hist   = []

print("\n=== Training final model ===")
for ep in range(1, BEST["epochs"]+1):
    # ---- train ----
    model.train(); run_loss = 0.0; y_true = []; y_pred = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(); out = model(xb)
        loss = criterion(out, yb); loss.backward(); optimizer.step()
        run_loss += loss.item() * yb.size(0)
        y_true.extend(yb.cpu().numpy()); y_pred.extend(torch.argmax(out,1).cpu().numpy())
    tr_loss = run_loss / len(y_train)
    tr_f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # ---- validate ----
    model.eval(); v_loss = 0.0; y_v, y_p = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            v_loss += criterion(out, yb).item() * yb.size(0)
            y_v.extend(yb.cpu().numpy()); y_p.extend(torch.argmax(out,1).cpu().numpy())
    v_loss /= len(y_val)
    v_f1    = f1_score(y_v, y_p, average="weighted", zero_division=0)

    train_loss_hist.append(tr_loss); val_loss_hist.append(v_loss)
    train_f1_hist  .append(tr_f1);  val_f1_hist  .append(v_f1)

    if v_loss < best_val_loss - 1e-4:
        best_val_loss, wait = v_loss, 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping at epoch {ep}"); break

    if ep % 10 == 0:
        print(f"[Epoch {ep:3d}] train_loss={tr_loss:.4f}  val_loss={v_loss:.4f}")

model.load_state_dict(torch.load("best_model.pt"))

# ---------------------------------------------------------------------
# 7. Learning curves (loss & F1)
# ---------------------------------------------------------------------

epochs_r = range(1, len(train_loss_hist)+1)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs_r, train_loss_hist, label="Train")
plt.plot(epochs_r, val_loss_hist,   label="Val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Learning Curve: Loss"); plt.legend()
plt.subplot(1,2,2)
plt.plot(epochs_r, train_f1_hist, label="Train")
plt.plot(epochs_r, val_f1_hist,   label="Val")
plt.xlabel("Epoch"); plt.ylabel("Weighted F1"); plt.title("Learning Curve: F1"); plt.legend()
plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------
# 8. K‑fold CV (k = 1…5) on training set
# ---------------------------------------------------------------------

def eval_fold(model, loader):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.extend(torch.argmax(model(xb.to(DEVICE)),1).cpu().numpy())
            trues.extend(yb.numpy())
    return accuracy_score(trues, preds)

k_list, acc_list = [], []
for k in range(1, 6):
    if k == 1:
        acc = eval_fold(model, val_loader)
    else:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
        acc_fold = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_va, y_va = X_train[va_idx], y_train[va_idx]
            tr_load = make_loader(X_tr, y_tr, batch_size=BEST["batch"])
            va_load = make_loader(X_va, y_va, batch_size=BEST["batch"], shuffle=False)
            net = FinalNet(BEST).to(DEVICE)
            opt = torch.optim.Adam(net.parameters(), lr=BEST["lr"])
            freq = np.bincount(y_tr)+1e-6
            w = torch.tensor(np.log(freq.sum()/freq), dtype=torch.float32, device=DEVICE)
            crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.15)
            for _ in range(10):
                net.train()
                for xb, yb in tr_load:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad(); l = crit(net(xb), yb); l.backward(); opt.step()
            acc_fold.append(eval_fold(net, va_load)); del net; torch.cuda.empty_cache(); gc.collect()
        acc = np.mean(acc_fold)
    k_list.append(k); acc_list.append(acc)
    print(f"k={k}  accuracy={acc:.3f}")

plt.figure(figsize=(6,4))
plt.plot(k_list, acc_list, marker="o"); plt.xticks(k_list)
plt.xlabel("Number of folds k"); plt.ylabel("Mean validation accuracy")
plt.title("Accuracy vs. k (K‑Fold CV)"); plt.tight_layout(); plt.show()

# ---------------------------------------------------------------------
# 8b. Accuracy vs number of training samples ─ Original vs Balanced
# ---------------------------------------------------------------------
print("\n=== Accuracy vs #Training Samples (Original vs Balanced) ===")
sizes_orig, acc_orig = acc_vs_samples(
    Xo_train, yo_train, Xo_val, yo_val, BEST, n_points=10, epochs=15)
sizes_bal,  acc_bal  = acc_vs_samples(
    X_train, y_train, X_val, y_val, BEST, n_points=10, epochs=15)

plt.figure(figsize=(7,5))
plt.plot(sizes_orig, acc_orig, marker="o", label="Original (unbalanced)")
plt.plot(sizes_bal,  acc_bal,  marker="s", label="Balanced (SMOTENC)")
plt.xlabel("Number of training samples")
plt.ylabel("Validation accuracy")
plt.title("Accuracy vs. Training Set Size")
plt.legend()
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 9. Final test evaluation
# ---------------------------------------------------------------------

test_loader = make_loader(X_test, y_test, batch_size=BEST["batch"], shuffle=False)
model.eval(); all_pred, all_true = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        all_pred.extend(torch.argmax(model(xb.to(DEVICE)),1).cpu().numpy())
        all_true.extend(yb.numpy())

print("\n=== Test classification report ===")
print(classification_report(all_true, all_pred, digits=4, zero_division=0))

cm = confusion_matrix(all_true, all_pred)
plt.figure(figsize=(8,6))
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.title("Confusion Matrix (Test)"); plt.tight_layout(); plt.show()

print("\n=== Key metrics (Test) ===")
print(f"Accuracy  : {accuracy_score(all_true, all_pred):.3f}")
print(f"Weighted F1: {f1_score(all_true, all_pred, average='weighted', zero_division=0):.3f}")

# ---------------------------------------------------------------------
# 10. Save trained model & config
# ---------------------------------------------------------------------
MODEL_PATH = "heart_disease_mlp_balanced.pkl"
joblib.dump({
    "model_state": model.state_dict(),
    "params": BEST,
    "input_dim": input_dim,
    "num_class": num_class,
}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
