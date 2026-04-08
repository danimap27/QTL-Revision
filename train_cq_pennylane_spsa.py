#!/usr/bin/env python3
"""
train_cq_pennylane_spsa.py -- PennyLane hybrid training with SPSA optimizer.

Addresses MAJOR reviewer issue #4: gradient methods confound framework comparison.
Instead of Adam with parameter-shift gradients (PennyLane's default), this script
uses SPSA (Simultaneous Perturbation Stochastic Approximation) which estimates
gradients with only 2 function evaluations per step -- matching Qiskit's approach
and enabling a fair framework comparison.

Usage:
    python train_cq_pennylane_spsa.py --dataset hymenoptera --model resnet18 --seed 42
    python train_cq_pennylane_spsa.py --dataset brain_tumor --model mobilenetv2 --spsa-lr 0.05
"""

import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
import random
import argparse
import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

import pennylane as qml

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset resolution (identical to other scripts)
# ---------------------------------------------------------------------------

def _resolve_dataset_dir(name: str):
    script_root = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    candidates = [
        os.path.join(script_root, 'Resultados', 'datasets', name),
        os.path.join(script_root, 'datasets', name),
        os.path.join(cwd, 'Resultados', 'datasets', name),
        os.path.join(cwd, 'datasets', name),
        os.path.join(script_root, 'user_datasets', name),
        os.path.join(cwd, 'user_datasets', name),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'train')) and os.path.isdir(os.path.join(c, 'test')):
            return c
    raise FileNotFoundError(
        f"Dataset '{name}' not found. Checked: " + ' | '.join(candidates)
    )


TRANSFORMS = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}


# ---------------------------------------------------------------------------
# SPSA Optimizer
# ---------------------------------------------------------------------------

class SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation optimizer.

    Estimates gradients using only 2 function evaluations per step (loss_plus
    and loss_minus) regardless of the number of parameters, matching Qiskit's
    SPSA implementation for fair framework comparison.

    References:
        Spall, J.C. (1998). Implementation of the simultaneous perturbation
        algorithm for stochastic optimization. IEEE Transactions on Aerospace
        and Electronic Systems, 34(3), 817-823.

    Args:
        params:        Iterable of torch Parameters to optimize.
        lr:            Base learning rate (a in Spall notation).
        perturbation:  Base perturbation magnitude (c in Spall notation).
        A:             Stability constant for learning rate schedule.
        alpha:         Exponent for learning rate decay (default 0.602).
        gamma:         Exponent for perturbation decay (default 0.101).
    """

    def __init__(self, params, lr=0.1, perturbation=0.1, A=10,
                 alpha=0.602, gamma=0.101):
        self.params = [p for p in params if p.requires_grad]
        self.a = lr
        self.c = perturbation
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.k = 0

    def step(self, loss_fn):
        """
        Perform one SPSA update step.

        Args:
            loss_fn: Callable that returns a scalar loss (no backward needed).

        Returns:
            Estimated loss at the current point (mean of loss_plus and loss_minus).
        """
        self.k += 1
        ak = self.a / (self.k + self.A) ** self.alpha
        ck = self.c / self.k ** self.gamma

        # Generate Bernoulli perturbation vectors for all parameters
        deltas = []
        for p in self.params:
            delta = torch.bernoulli(torch.ones_like(p.data) * 0.5) * 2 - 1
            deltas.append(delta)

        # Perturb +
        for p, delta in zip(self.params, deltas):
            p.data.add_(ck * delta)

        with torch.no_grad():
            loss_plus = loss_fn().item()

        # Perturb - (go from +ck*delta to -ck*delta)
        for p, delta in zip(self.params, deltas):
            p.data.sub_(2 * ck * delta)

        with torch.no_grad():
            loss_minus = loss_fn().item()

        # Restore original parameters and apply gradient estimate
        for p, delta in zip(self.params, deltas):
            p.data.add_(ck * delta)  # restore to original
            grad_est = (loss_plus - loss_minus) / (2 * ck * delta)
            p.data.sub_(ak * grad_est)

        return (loss_plus + loss_minus) / 2.0

    def zero_grad(self):
        """No-op for API compatibility (SPSA does not use autograd)."""
        pass


# ---------------------------------------------------------------------------
# Quantum layer (same architecture as train_cq_pennylane.py)
# ---------------------------------------------------------------------------

def create_quantum_layer(n_qubits, n_outputs, depth):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch')
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_outputs)]

    weight_shapes = {'weights': (depth, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class HybridModel(nn.Module):
    def __init__(self, backbone, feat_dim, n_qubits, n_outputs, depth=3):
        super().__init__()
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.fc = nn.Linear(feat_dim, n_qubits)
        self.qlayer = create_quantum_layer(n_qubits, n_outputs, depth)

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.tanh(x) * (np.pi / 2)
        if x.dim() == 1:
            return self.qlayer(x)
        outs = [self.qlayer(xi) for xi in x]
        return torch.stack(outs, dim=0)


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_pennylane_spsa(
    dataset_file="hymenoptera",
    classical_model="resnet18",
    n_qubits=4,
    quantum_depth=3,
    epochs=10,
    batch_size=16,
    seed=42,
    spsa_lr=0.1,
    spsa_perturbation=0.1,
    spsa_A=10,
    early_stop_patience=10,
    run_id="null",
    output_dir=".",
):
    """Train a PennyLane hybrid model using SPSA optimizer."""

    print("=" * 70)
    print("PennyLane Quantum Transfer Learning -- SPSA Optimizer")
    print("=" * 70)
    print(f"  Dataset       : {dataset_file}")
    print(f"  Model         : {classical_model}")
    print(f"  Qubits        : {n_qubits}")
    print(f"  Depth         : {quantum_depth}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Seed          : {seed}")
    print(f"  SPSA lr       : {spsa_lr}")
    print(f"  SPSA perturb  : {spsa_perturbation}")
    print(f"  SPSA A        : {spsa_A}")
    print(f"  Run ID        : {run_id}")
    print(f"  Output dir    : {output_dir}")
    print("=" * 70)

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("Step 1/7: Loading dataset...")
    data_dir = _resolve_dataset_dir(dataset_file)
    print(f"  Dataset path: {data_dir}")

    full_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), TRANSFORMS['train'])
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'), TRANSFORMS['test'])
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    num_classes = len(full_train.classes)
    class_names = full_train.classes
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}, Classes: {num_classes}")

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------
    print("Step 2/7: Loading classical backbone...")
    if classical_model.lower() == 'resnet18':
        backbone = models.resnet18(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif classical_model.lower() == 'resnet34':
        backbone = models.resnet34(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif classical_model.lower() == 'mobilenetv2':
        backbone = models.mobilenet_v2(pretrained=True)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    elif classical_model.lower() == 'efficientnet_b0':
        backbone = models.efficientnet_b0(pretrained=True)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    elif classical_model.lower() == 'regnet_x_400mf':
        backbone = models.regnet_x_400mf(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        raise ValueError(
            "Unsupported model. Use 'resnet18', 'resnet34', 'mobilenetv2', "
            "'efficientnet_b0', or 'regnet_x_400mf'."
        )
    backbone = backbone.to(device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("Step 3/7: Creating hybrid quantum model...")
    model = HybridModel(backbone, feat_dim, n_qubits, num_classes, quantum_depth).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params}")

    # ------------------------------------------------------------------
    # SPSA Optimizer
    # ------------------------------------------------------------------
    print("Step 4/7: Initializing SPSA optimizer...")
    criterion = nn.CrossEntropyLoss()
    spsa_optimizer = SPSAOptimizer(
        params=model.parameters(),
        lr=spsa_lr,
        perturbation=spsa_perturbation,
        A=spsa_A,
    )
    print(f"  SPSA parameters: a={spsa_lr}, c={spsa_perturbation}, A={spsa_A}")

    # CodeCarbon
    energy_kwh = 0.0
    co2_kg = 0.0
    tracker = None
    if HAS_CODECARBON:
        try:
            print("  Initializing CodeCarbon energy tracker...")
            tracker = EmissionsTracker(
                project_name=f"pennylane_spsa_{run_id}",
                log_level="error",
                save_to_file=False,
            )
            tracker.start()
        except Exception:
            print("  CodeCarbon initialization failed -- energy tracking disabled.")
            tracker = None
    else:
        print("  CodeCarbon not available -- energy tracking disabled.")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("Step 5/7: Training with SPSA...")
    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, val_accs = [], [], []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # SPSA step: define loss_fn that evaluates the model at current params
            def loss_fn():
                out = model(x)
                return criterion(out, y)

            batch_loss = spsa_optimizer.step(loss_fn)
            epoch_loss += batch_loss
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item() * x.size(0)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_loss = v_loss / total
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time

    if tracker is not None:
        try:
            emissions = tracker.stop()
            if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                energy_kwh = float(tracker.final_emissions_data.energy_consumed)
                co2_kg = float(tracker.final_emissions_data.emissions)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("Step 6/7: Evaluating on test set...")
    model.load_state_dict(best_state)
    model.eval()

    y_true, y_pred, y_scores = [], [], []
    test_start = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
    test_time = time.time() - test_start

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)

    test_acc = (y_pred == y_true).mean()
    prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    if num_classes == 2:
        auc_roc_w = roc_auc_score(y_true, y_scores[:, 1])
    else:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        auc_roc_w = roc_auc_score(y_true_bin, y_scores, average='weighted', multi_class='ovr')

    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Precision     : {prec_w:.4f}")
    print(f"  Recall        : {rec_w:.4f}")
    print(f"  F1            : {f1_w:.4f}")
    print(f"  AUC-ROC       : {auc_roc_w:.4f}")
    print(f"  Train time    : {train_time:.2f}s")
    print(f"  Energy (kWh)  : {energy_kwh:.6f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    print("Step 7/7: Saving model, metrics, and CSV...")

    # Model checkpoint
    model_dir = os.path.join(output_dir, 'model_saved')
    os.makedirs(model_dir, exist_ok=True)
    model_fn = f"PL_SPSA_{run_id}.pth"
    torch.save(model.state_dict(), os.path.join(model_dir, model_fn))
    print(f"  Model saved: {os.path.join(model_dir, model_fn)}")

    # Metrics plots
    metrics_dir = os.path.join(output_dir, 'static', 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    plt.style.use('default')
    sns.set_palette("husl")

    # Loss plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2,
             marker='o', markersize=4, label='Train Loss (SPSA)')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', linewidth=2,
             marker='s', markersize=4, label='Val Loss')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Loss vs Epoch -- PennyLane SPSA {classical_model.upper()}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{run_id}_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, len(val_accs) + 1), [a * 100 for a in val_accs], 'g-',
             linewidth=2, marker='d', markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy (%)", fontsize=12)
    plt.title(f"Val Accuracy vs Epoch -- PennyLane SPSA {classical_model.upper()}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{run_id}_acc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix -- PennyLane SPSA {classical_model.upper()}", fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{run_id}_confmat.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC curves
    plt.figure(figsize=(10, 8), dpi=300)
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc_val:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
    else:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        for i in range(num_classes):
            fpr_i, tpr_i, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc_val = auc(fpr_i, tpr_i)
            plt.plot(fpr_i, tpr_i, lw=2,
                     label=f'{class_names[i]} (AUC = {roc_auc_val:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve -- PennyLane SPSA {classical_model.upper()}', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{run_id}_roc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # CSV output
    csv_dir = output_dir
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"PL_SPSA_{run_id}.csv")
    header = (
        "approach,backbone,dataset,seed,n_qubits,quantum_depth,optimizer,"
        "spsa_lr,spsa_perturbation,spsa_A,"
        "test_accuracy,precision_weighted,recall_weighted,f1_weighted,"
        "auc_roc_weighted,train_time_s,test_time_s,energy_kwh,co2_kg,"
        "epochs_actual,trainable_params,loss_history,val_acc_history"
    )
    row = (
        f"PL_SPSA,{classical_model},{dataset_file},{seed},{n_qubits},{quantum_depth},SPSA,"
        f"{spsa_lr},{spsa_perturbation},{spsa_A},"
        f"{test_acc:.6f},{prec_w:.6f},{rec_w:.6f},{f1_w:.6f},"
        f"{auc_roc_w:.6f},{train_time:.2f},{test_time:.2f},{energy_kwh:.8f},{co2_kg:.8f},"
        f"{len(train_losses)},{trainable_params},"
        f"\"{json.dumps([round(v, 6) for v in train_losses])}\","
        f"\"{json.dumps([round(v, 6) for v in val_accs])}\""
    )
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header + "\n")
        f.write(row + "\n")
    print(f"  CSV saved: {csv_path}")

    # Summary
    print()
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"  Optimizer     : SPSA (a={spsa_lr}, c={spsa_perturbation}, A={spsa_A})")
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Precision     : {prec_w:.4f}")
    print(f"  Recall        : {rec_w:.4f}")
    print(f"  F1            : {f1_w:.4f}")
    print(f"  AUC-ROC       : {auc_roc_w:.4f}")
    print(f"  Training Time : {train_time:.2f}s")
    print(f"  Testing Time  : {test_time:.2f}s")
    print(f"  Energy (kWh)  : {energy_kwh:.6f}")
    print("=" * 70)

    return {
        'test_accuracy': test_acc,
        'precision': prec_w,
        'recall': rec_w,
        'f1': f1_w,
        'auc_roc': auc_roc_w,
        'train_time': train_time,
        'test_time': test_time,
        'energy_kwh': energy_kwh,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Train a PennyLane hybrid model using SPSA optimizer "
            "(gradient-free, fair comparison with Qiskit)."
        ),
    )
    p.add_argument('--dataset', default='hymenoptera',
                   help='Dataset name (must have train/ and test/ subdirs)')
    p.add_argument('--model', default='resnet18',
                   choices=['resnet18', 'resnet34', 'mobilenetv2',
                            'efficientnet_b0', 'regnet_x_400mf'],
                   help='Pre-trained backbone architecture')
    p.add_argument('--n-qubits', type=int, default=4,
                   help='Number of qubits (default: 4)')
    p.add_argument('--depth', type=int, default=3,
                   help='Quantum circuit depth (default: 3)')
    p.add_argument('--epochs', type=int, default=10,
                   help='Number of training epochs (default: 10)')
    p.add_argument('--batch-size', type=int, default=16,
                   help='Batch size (default: 16)')
    p.add_argument('--patience', type=int, default=10,
                   help='Early stopping patience (default: 10)')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility (default: 42)')

    # SPSA-specific parameters
    p.add_argument('--spsa-lr', type=float, default=0.1,
                   help='SPSA base learning rate (a parameter, default: 0.1)')
    p.add_argument('--spsa-perturbation', type=float, default=0.1,
                   help='SPSA perturbation magnitude (c parameter, default: 0.1)')
    p.add_argument('--spsa-A', type=float, default=10,
                   help='SPSA stability constant (default: 10)')

    p.add_argument('--output-dir', default='.',
                   help='Base output directory (default: current dir)')
    p.add_argument('--id', default=None,
                   help='Run identifier (auto-generated if not provided)')

    args = p.parse_args()

    # Auto-generate ID
    if args.id is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.id = (
            f"spsa_{args.model}_{args.n_qubits}q_depth{args.depth}"
            f"_{args.dataset}_s{args.seed}_{ts}"
        )

    try:
        result = train_pennylane_spsa(
            dataset_file=args.dataset,
            classical_model=args.model,
            n_qubits=args.n_qubits,
            quantum_depth=args.depth,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            spsa_lr=args.spsa_lr,
            spsa_perturbation=args.spsa_perturbation,
            spsa_A=args.spsa_A,
            early_stop_patience=args.patience,
            run_id=args.id,
            output_dir=args.output_dir,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
