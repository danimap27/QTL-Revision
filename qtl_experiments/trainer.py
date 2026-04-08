"""Generic training loop for all head types."""

import os
import time
import csv
import traceback
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

from heads import create_head
from data import load_dataset


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

BACKBONE_REGISTRY = {
    "resnet18":       (models.resnet18,       "fc",         512),
    "resnet34":       (models.resnet34,       "fc",         512),
    "mobilenetv2":    (models.mobilenet_v2,   "classifier", 1280),
    "efficientnet_b0":(models.efficientnet_b0,"classifier", 1280),
    "regnet_x_400mf": (models.regnet_x_400mf,"fc",         400),
}


def _load_backbone(name):
    """Load a pretrained backbone and replace its classifier with Identity."""
    if name not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: {name}")
    factory, clf_attr, feat_dim = BACKBONE_REGISTRY[name]
    try:
        model = factory(weights="DEFAULT")
    except TypeError:
        model = factory(pretrained=True)

    # Remove classifier
    if clf_attr == "fc":
        model.fc = nn.Identity()
    elif clf_attr == "classifier":
        if hasattr(model.classifier, '__len__') and len(model.classifier) > 1:
            model.classifier[-1] = nn.Identity()
        else:
            model.classifier = nn.Identity()

    for p in model.parameters():
        p.requires_grad = False
    return model, feat_dim


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CSV writers (append mode, thread-safe via file locking not needed for sequential)
# ---------------------------------------------------------------------------

def _append_csv(filepath, row_dict, fieldnames):
    """Append a single row to a CSV, creating with headers if needed."""
    write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    with open(filepath, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


RUNS_FIELDS = [
    "run_id", "seed", "dataset", "backbone", "head", "head_type",
    "n_qubits", "depth", "shots", "n_trainable_params", "epochs",
    "lr", "batch_size", "train_time_s", "energy_kwh",
    "test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc",
    "timestamp",
]

PREDICTIONS_FIELDS = [
    "run_id", "sample_idx", "y_true", "y_pred", "y_prob_0", "y_prob_1",
]

TRAINING_LOG_FIELDS = [
    "run_id", "epoch", "train_loss", "val_loss", "val_accuracy", "epoch_time_s",
]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

class _SPSAOptimizer:
    """SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

    Gradient-free optimizer for fair comparison with Qiskit's default SPSA.
    Uses Spall's gain sequences: a_k = a/(k+A)^alpha, c_k = c/(k+1)^gamma.
    """

    def __init__(self, params, lr=0.1, a=0.1, c=0.01, A=10, alpha=0.602, gamma_spsa=0.101):
        self.params = list(params)
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma_spsa = gamma_spsa
        self.k = 0

    def step_spsa(self, model, criterion, x, y):
        """One SPSA step: perturb params, evaluate +/-, estimate gradient, update."""
        self.k += 1
        ak = self.a / (self.k + self.A) ** self.alpha
        ck = self.c / (self.k + 1) ** self.gamma_spsa

        # Generate perturbation
        deltas = []
        for p in self.params:
            if p.requires_grad:
                deltas.append(torch.bernoulli(torch.full_like(p.data, 0.5)) * 2 - 1)
            else:
                deltas.append(torch.zeros_like(p.data))

        # Perturb +
        for p, d in zip(self.params, deltas):
            p.data.add_(d * ck)
        with torch.no_grad():
            loss_plus = criterion(model(x), y).item()

        # Perturb - (total shift = -2*ck*delta)
        for p, d in zip(self.params, deltas):
            p.data.sub_(2 * d * ck)
        with torch.no_grad():
            loss_minus = criterion(model(x), y).item()

        # Restore original and apply gradient estimate
        for p, d in zip(self.params, deltas):
            p.data.add_(d * ck)  # back to original
            grad_est = (loss_plus - loss_minus) / (2.0 * ck * d.clamp(min=1e-10))
            p.data.sub_(ak * grad_est)

        return (loss_plus + loss_minus) / 2.0


class HybridModel(nn.Module):
    """Backbone (frozen) + Head (trainable)."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        if features.dim() > 2:
            features = torch.flatten(features, 1)
        return self.head(features)


def train_and_evaluate(run_cfg, config):
    """Execute one complete training run.

    Args:
        run_cfg: dict with keys run_id, seed, dataset (cfg dict),
                 backbone (cfg dict), head (cfg dict)
        config: full parsed config dict

    Returns:
        dict with all results, or raises on failure
    """
    run_id = run_cfg["run_id"]
    seed = run_cfg["seed"]
    dataset_cfg = run_cfg["dataset"]
    backbone_cfg = run_cfg["backbone"]
    head_cfg = run_cfg["head"]
    training_cfg = config["training"]
    output_dir = config.get("output_dir", "./results")
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints")

    os.makedirs(output_dir, exist_ok=True)

    # Seed
    set_seed(seed)

    # Device
    dev_str = config.get("device", "auto")
    if dev_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(dev_str)

    # Data
    train_loader, val_loader, test_loader, num_classes = load_dataset(
        dataset_cfg, training_cfg, seed
    )

    # Backbone
    backbone, feat_dim = _load_backbone(backbone_cfg["name"])
    backbone = backbone.to(device)

    # Head
    head, head_type = create_head(head_cfg, feat_dim, num_classes)
    head = head.to(device)

    # Model
    model = HybridModel(backbone, head).to(device)

    # Count trainable params (head only)
    n_trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)

    # Optimizer & scheduler
    lr = training_cfg.get("lr", 0.001)
    opt_name = head_cfg.get("optimizer_override", training_cfg.get("optimizer", "adam"))
    use_spsa = (opt_name == "spsa")

    if use_spsa:
        # SPSA: gradient-free optimizer (M1 reviewer control)
        optimizer = _SPSAOptimizer(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, a=0.1, c=0.01,
        )
        scheduler = None
    else:
        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
        sched_cfg = training_cfg.get("scheduler", {})
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=sched_cfg.get("step_size", 3),
            gamma=sched_cfg.get("gamma", 0.9),
        )
    criterion = nn.CrossEntropyLoss()
    epochs = training_cfg.get("epochs", 10)

    # Checkpoint dir for this run
    run_ckpt_dir = os.path.join(checkpoint_dir, run_id)
    os.makedirs(run_ckpt_dir, exist_ok=True)

    # Check for resume from checkpoint
    start_epoch = 1
    ckpt_files = sorted([f for f in os.listdir(run_ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt")])
    if ckpt_files:
        last_ckpt = os.path.join(run_ckpt_dir, ckpt_files[-1])
        ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if not use_spsa:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"  Resuming from epoch {start_epoch}")

    # Energy tracking
    energy_kwh = None
    tracker = None
    energy_cfg = config.get("energy", {})
    if energy_cfg.get("enabled") and energy_cfg.get("tool") == "codecarbon":
        try:
            from codecarbon import EmissionsTracker
            tracker = EmissionsTracker(
                project_name=run_id,
                output_dir=os.path.join(output_dir, "energy"),
                log_level="error",
            )
            tracker.start()
        except Exception:
            pass

    # ---- Training loop ----
    t0 = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        ep_start = time.perf_counter()

        # Train
        model.train()
        running_loss = 0.0
        n_samples = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if use_spsa:
                loss = optimizer.step_spsa(model, criterion, x, y)
                running_loss += loss * x.size(0)
            else:
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
            n_samples += x.size(0)
        train_loss = running_loss / n_samples

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss_sum += loss.item() * x.size(0)
                _, preds = torch.max(out, 1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        if scheduler:
            scheduler.step()
        ep_time = time.perf_counter() - ep_start

        # Log to training_log.csv
        _append_csv(
            os.path.join(output_dir, "training_log.csv"),
            {
                "run_id": run_id, "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_accuracy": f"{val_acc:.6f}",
                "epoch_time_s": f"{ep_time:.2f}",
            },
            TRAINING_LOG_FIELDS,
        )

        # Save checkpoint
        if config.get("checkpoints", {}).get("save_every_epoch", True):
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                os.path.join(run_ckpt_dir, f"epoch_{epoch:03d}.pt"),
            )

        print(f"  Epoch {epoch}/{epochs} — train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({ep_time:.1f}s)")

    train_time = time.perf_counter() - t0

    # Energy stop
    if tracker:
        try:
            emissions = tracker.stop()
            energy_kwh = tracker._total_energy.kWh if hasattr(tracker, "_total_energy") else 0.0
        except Exception:
            energy_kwh = None

    # ---- Test evaluation ----
    model.eval()
    all_y_true, all_y_pred, all_y_prob = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(out, 1)
            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs.cpu().numpy())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        if num_classes == 2:
            auc_val = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc_val = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        auc_val = 0.0

    # ---- Write predictions.csv ----
    pred_path = os.path.join(output_dir, "predictions.csv")
    for idx in range(len(y_true)):
        _append_csv(
            pred_path,
            {
                "run_id": run_id,
                "sample_idx": idx,
                "y_true": int(y_true[idx]),
                "y_pred": int(y_pred[idx]),
                "y_prob_0": f"{y_prob[idx, 0]:.6f}",
                "y_prob_1": f"{y_prob[idx, 1]:.6f}" if y_prob.shape[1] > 1 else "0.0",
            },
            PREDICTIONS_FIELDS,
        )

    # ---- Write runs.csv ----
    n_qubits = head_cfg.get("n_qubits", None)
    depth = head_cfg.get("depth", None)
    shots = head_cfg.get("shots", None)

    _append_csv(
        os.path.join(output_dir, "runs.csv"),
        {
            "run_id": run_id,
            "seed": seed,
            "dataset": dataset_cfg["name"],
            "backbone": backbone_cfg["name"],
            "head": head_cfg["name"],
            "head_type": head_type,
            "n_qubits": n_qubits if n_qubits else "",
            "depth": depth if depth else "",
            "shots": shots if shots else "",
            "n_trainable_params": n_trainable,
            "epochs": epochs,
            "lr": lr,
            "batch_size": training_cfg["batch_size"],
            "train_time_s": f"{train_time:.2f}",
            "energy_kwh": f"{energy_kwh:.6f}" if energy_kwh is not None else "",
            "test_accuracy": f"{acc:.6f}",
            "test_precision": f"{prec:.6f}",
            "test_recall": f"{rec:.6f}",
            "test_f1": f"{f1:.6f}",
            "test_auc": f"{auc_val:.6f}",
            "timestamp": datetime.now().isoformat(),
        },
        RUNS_FIELDS,
    )

    # Save final model
    if config.get("checkpoints", {}).get("save_final_model", True):
        models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(models_dir, f"{run_id}.pt"))

    return {
        "run_id": run_id,
        "test_accuracy": acc,
        "train_time_s": train_time,
        "n_trainable_params": n_trainable,
    }
