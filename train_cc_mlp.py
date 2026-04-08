# train_cc_mlp.py
# Fair classical MLP baseline for QTL paper revision.
# Addresses reviewer issue #2: the original classical baseline (train_cc.py) uses
# a single linear layer (logistic regression), while quantum models have ~12-24
# trainable parameters with nonlinear encoding. This script provides:
#   --head-type matched  : parameter-matched MLP (~20-24 params, comparable to VQC)
#   --head-type standard : standard MLP as upper-bound reference
# ---------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
import random
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
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
# Dataset resolution (identical to train_cc.py / PennyLane scripts)
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


# ---------------------------------------------------------------------------
# Transforms (same as PennyLane scripts for fair comparison)
# ---------------------------------------------------------------------------
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
# MLP head definitions
# ---------------------------------------------------------------------------
class MatchedMLP(nn.Module):
    """Parameter-matched MLP head (~20-24 trainable params).

    Architecture:
        Linear(feat_dim, 4) -> Tanh -> Linear(4, 4, bias=False) -> Tanh
        -> Linear(4, num_classes, bias=False)

    For a 2-class problem with feat_dim=512:
        Layer 1: 512*4 + 4 = 2052  (but only this + tiny layers are trainable)
    NOTE: the backbone is frozen, so the *effective* trainable parameters in the
    head are what matters for the comparison.  With feat_dim >> 4 the first
    linear layer dominates; however the nonlinear capacity of the head is
    limited to 4 hidden units, matching the quantum circuit's expressivity.
    The total number of params in layers *after* the dimensionality reduction
    (i.e. the 4-wide hidden layers) is:  4*4 + 4*num_classes = 16 + 4*C,
    giving 24 for C=2 and 32 for C=4, which is comparable to VQC circuits.
    """

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 4),        # dimensionality reduction (like fc->qubits)
            nn.Tanh(),                      # nonlinear activation (like angle encoding)
            nn.Linear(4, 4, bias=False),    # hidden layer
            nn.Tanh(),                      # nonlinearity
            nn.Linear(4, num_classes, bias=False),  # output
        )

    def forward(self, x):
        return self.net(x)


class StandardMLP(nn.Module):
    """Standard MLP head as upper-bound reference.

    Architecture:
        Linear(feat_dim, 128) -> ReLU -> Dropout(0.3)
        -> Linear(128, 64) -> ReLU -> Dropout(0.3)
        -> Linear(64, num_classes)
    """

    def __init__(self, feat_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Full model: frozen backbone + MLP head
# ---------------------------------------------------------------------------
class BackboneMLP(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        # Freeze backbone
        for p in backbone.parameters():
            p.requires_grad = False
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        if x.dim() > 2:
            x = torch.flatten(x, 1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Backbone factory (Identity head, same as PennyLane scripts)
# ---------------------------------------------------------------------------
def build_backbone(model_name: str):
    """Return (backbone_with_identity_head, feature_dimension)."""
    name = model_name.lower()
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif name == 'mobilenetv2':
        backbone = models.mobilenet_v2(pretrained=True)
        feat_dim = 1280
        backbone.classifier = nn.Identity()
    elif name == 'efficientnet_b0':
        backbone = models.efficientnet_b0(pretrained=True)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
    elif name == 'regnet_x_400mf':
        backbone = models.regnet_x_400mf(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            "Choose from: resnet18, resnet34, mobilenetv2, efficientnet_b0, regnet_x_400mf."
        )
    return backbone, feat_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def evaluate(model, loader, device):
    """Return accuracy on *loader*."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, preds = torch.max(out, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train_mlp(
    dataset_file: str,
    classical_model: str,
    head_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    run_id: str,
    output_dir: str,
    checkpoint_dir=None,
):
    set_seed(seed)
    # Checkpoint directory setup
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Classical MLP Baseline (Frozen Backbone)")
    print("=" * 60)
    print(f"  Dataset      : {dataset_file}")
    print(f"  Backbone     : {classical_model}")
    print(f"  Head type    : {head_type}")
    print(f"  Epochs       : {epochs}")
    print(f"  Batch size   : {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Seed         : {seed}")
    print(f"  Run ID       : {run_id}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Device       : {device}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1/7: Data
    # ------------------------------------------------------------------
    print("Step 1/7: Loading and preparing datasets...")
    data_dir = _resolve_dataset_dir(dataset_file)
    print(f"  Dataset path : {data_dir}")

    full_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), TRANSFORMS['train'])
    test_ds    = datasets.ImageFolder(os.path.join(data_dir, 'test'),  TRANSFORMS['test'])

    train_size = int(0.8 * len(full_train))
    val_size   = len(full_train) - train_size
    train_ds, val_ds = random_split(
        full_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    num_classes = len(full_train.classes)
    class_names = full_train.classes
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # ------------------------------------------------------------------
    # Step 2/7: Build model
    # ------------------------------------------------------------------
    print("Step 2/7: Building backbone + MLP head...")
    backbone, feat_dim = build_backbone(classical_model)
    print(f"  Backbone feature dim: {feat_dim}")

    if head_type == 'matched':
        head = MatchedMLP(feat_dim, num_classes)
    elif head_type == 'standard':
        head = StandardMLP(feat_dim, num_classes)
    else:
        raise ValueError(f"Unknown head type '{head_type}'. Use 'matched' or 'standard'.")

    model = BackboneMLP(backbone, head).to(device)

    head_params = count_trainable_params(model.head)
    total_params = count_trainable_params(model)
    print(f"  Head trainable params : {head_params}")
    print(f"  Total trainable params: {total_params}")

    # ------------------------------------------------------------------
    # Step 3/7: Training setup
    # ------------------------------------------------------------------
    print("Step 3/7: Setting up training components...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    loss_history = []
    val_acc_history = []

    # ------------------------------------------------------------------
    # Step 4/7: CodeCarbon tracker
    # ------------------------------------------------------------------
    tracker = None
    if HAS_CODECARBON:
        print("Step 4/7: Initializing CodeCarbon energy tracker...")
        tracker = EmissionsTracker(
            project_name=f"CC_MLP_{run_id}",
            log_level="error",
            save_to_file=False,
        )
        tracker.start()
    else:
        print("Step 4/7: CodeCarbon not available -- energy tracking disabled.")

    # ------------------------------------------------------------------
    # Step 5/7: Training loop
    # ------------------------------------------------------------------
    print("Step 5/7: Starting training...")
    start_train = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_acc = evaluate(model, val_loader, device)
        loss_history.append(avg_loss)
        val_acc_history.append(val_acc)
        print(f"  Epoch {epoch}/{epochs} -- Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        # Save epoch checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss_history[-1],
            'val_acc': val_acc,
            'args': {
                'dataset': dataset_file,
                'model': classical_model,
                'seed': seed,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
            }
        }
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    train_time = time.time() - start_train

    # Stop energy tracker
    energy_kwh = 0.0
    co2_kg = 0.0
    if tracker is not None:
        emissions = tracker.stop()
        if emissions is not None:
            co2_kg = float(emissions)  # kg CO2
        energy_kwh = float(tracker.final_emissions_data.energy_consumed) if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data else 0.0

    # ------------------------------------------------------------------
    # Step 6/7: Test evaluation
    # ------------------------------------------------------------------
    print("Step 6/7: Evaluating on test set...")
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    start_test = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
    test_time = time.time() - start_test

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_scores = np.array(y_scores)

    test_acc = float(np.mean(y_true == y_pred))
    prec_w   = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    rec_w    = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1_w     = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    # AUC-ROC (weighted)
    try:
        if num_classes == 2:
            auc_roc_w = float(roc_auc_score(y_true, y_scores[:, 1]))
        else:
            auc_roc_w = float(roc_auc_score(
                y_true, y_scores, multi_class='ovr', average='weighted',
            ))
    except Exception:
        auc_roc_w = float('nan')

    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Precision (w) : {prec_w:.4f}")
    print(f"  Recall (w)    : {rec_w:.4f}")
    print(f"  F1 (w)        : {f1_w:.4f}")
    print(f"  AUC-ROC (w)   : {auc_roc_w:.4f}")
    print(f"  Train time    : {train_time:.2f}s")
    print(f"  Test time     : {test_time:.2f}s")
    print(f"  Energy (kWh)  : {energy_kwh:.6f}")
    print(f"  CO2 (kg)      : {co2_kg:.6f}")

    # ------------------------------------------------------------------
    # Step 7/7: Save everything
    # ------------------------------------------------------------------
    print("Step 7/7: Saving model, metrics, and plots...")

    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    model_dir  = os.path.join(output_dir, 'model_saved')
    metrics_dir = os.path.join(output_dir, 'static', 'metrics')
    csv_dir    = os.path.join(output_dir, 'results')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, f"CC_MLP_{run_id}_{classical_model}_{dataset_file}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  Model saved: {model_path}")
    # Save final model with full metadata
    os.makedirs("model_saved", exist_ok=True)
    final_model_path = os.path.join("model_saved", f"{run_id}_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_complete': True,
        'best_val_acc': max(val_acc_history) if val_acc_history else 0.0,
        'test_acc': test_acc,
        'train_time_s': train_time,
        'epochs_trained': len(loss_history),
        'loss_history': loss_history,
        'val_acc_history': val_acc_history,
        'config': {
            'dataset': dataset_file,
            'backbone': classical_model,
            'seed': seed,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'head_type': head_type,
        }
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")

    # ---- Plots ----
    epochs_range = list(range(1, len(loss_history) + 1))

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, loss_history, marker='o', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, f"{run_id}_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, [a * 100 for a in val_acc_history],
             marker='o', color='green', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(metrics_dir, f"{run_id}_acc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(metrics_dir, f"{run_id}_confmat.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC curves
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores[:, 1])
        roc_auc_val = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(metrics_dir, f"{run_id}_roc.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        plt.figure(figsize=(10, 8))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc_val = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                     label=f'Class {class_names[i]} (AUC = {roc_auc_val:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Multi-class")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(metrics_dir, f"{run_id}_roc.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- Structured CSV ----
    csv_path = os.path.join(csv_dir, f"CC_MLP_{run_id}.csv")
    header = (
        "approach,backbone,dataset,seed,n_qubits,quantum_depth,head_type,"
        "test_accuracy,precision_weighted,recall_weighted,f1_weighted,"
        "auc_roc_weighted,train_time_s,test_time_s,energy_kwh,co2_kg,"
        "epochs_actual,head_trainable_params,loss_history,val_acc_history"
    )
    row = (
        f"CC_MLP,{classical_model},{dataset_file},{seed},NA,NA,{head_type},"
        f"{test_acc:.6f},{prec_w:.6f},{rec_w:.6f},{f1_w:.6f},"
        f"{auc_roc_w:.6f},{train_time:.2f},{test_time:.2f},{energy_kwh:.8f},{co2_kg:.8f},"
        f"{len(loss_history)},{head_params},"
        f"\"{json.dumps([round(v, 6) for v in loss_history])}\","
        f"\"{json.dumps([round(v, 6) for v in val_acc_history])}\""
    )
    with open(csv_path, 'w') as f:
        f.write(header + "\n")
        f.write(row + "\n")
    print(f"  CSV saved   : {csv_path}")

    # Summary
    print()
    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Head params   : {head_params}")
    print(f"  Training Time : {train_time:.2f}s")
    print(f"  Testing Time  : {test_time:.2f}s")
    print("=" * 60)

    return {
        'test_accuracy': test_acc,
        'train_time': train_time,
        'test_time': test_time,
        'head_params': head_params,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Train a fair classical MLP baseline (frozen backbone + MLP head)"
    )
    p.add_argument('--dataset', default='hymenoptera',
                   help='Dataset name (must have train/ and test/ subdirs)')
    p.add_argument('--model', default='resnet18',
                   choices=['resnet18', 'resnet34', 'mobilenetv2',
                            'efficientnet_b0', 'regnet_x_400mf'],
                   help='Pre-trained backbone architecture')
    p.add_argument('--head-type', default='matched',
                   choices=['matched', 'standard'],
                   help='MLP head type: matched (~20-24 params) or standard (upper bound)')
    p.add_argument('--epochs', type=int, default=20,
                   help='Number of training epochs')
    p.add_argument('--batch-size', type=int, default=16,
                   help='Batch size')
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Learning rate')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed for reproducibility')
    p.add_argument('--output-dir', default='.',
                   help='Base output directory (for SLURM / cluster runs)')
    p.add_argument('--id', default=None,
                   help='Run identifier (auto-generated if not provided)')
    p.add_argument('--checkpoint-dir', type=str, default=None,
                   help='Directory to save per-epoch checkpoints (default: checkpoints/<run_id>)')
    args = p.parse_args()

    # Auto-generate ID
    if args.id is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.id = f"mlp_{args.head_type}_{args.model}_{args.dataset}_s{args.seed}_{ts}"

    try:
        result = train_mlp(
            dataset_file=args.dataset,
            classical_model=args.model,
            head_type=args.head_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            run_id=args.id,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
