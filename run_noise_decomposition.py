#!/usr/bin/env python3
"""
run_noise_decomposition.py -- Structured noise decomposition analysis.

Addresses MAJOR reviewer issue #10: noise analysis is too descriptive.
Instead of just reporting full-noise results, this script isolates the
impact of each noise component by testing 5 configurations:

    1. ideal            -- No noise (baseline)
    2. amplitude_only   -- Only amplitude damping
    3. phase_only       -- Only phase damping
    4. depolarizing_only-- Only depolarizing noise
    5. full_noise       -- All noise components (realistic IBM)

Fixed parameters: ResNet-18, Hymenoptera, 4 qubits, depth 3.
Seeds: [42, 123, 456, 789, 1024]

Total runs: 5 configs x 5 seeds = 25.
With SLURM: --task-id N  (N in 0..24)

Usage examples:
    # SLURM array job
    python run_noise_decomposition.py --task-id $SLURM_ARRAY_TASK_ID

    # Run everything sequentially
    python run_noise_decomposition.py --run-all

    # Resume interrupted run
    python run_noise_decomposition.py --run-all --resume

    # Dry run
    python run_noise_decomposition.py --run-all --dry-run
"""

import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

import argparse
import csv
import datetime
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

import pennylane as qml

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

NOISE_CONFIGS = [
    "ideal",
    "amplitude_only",
    "phase_only",
    "depolarizing_only",
    "full_noise",
]

SEEDS = [42, 123, 456, 789, 1024]

# Fixed experimental parameters
BACKBONE = "resnet18"
DATASET = "hymenoptera"
N_QUBITS = 4
DEPTH = 3

TOTAL_EXPERIMENTS = len(NOISE_CONFIGS) * len(SEEDS)  # 25

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
# Dataset resolution
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
# IBM noise parameters (from train_cq_pennylane_noisy.py)
# ---------------------------------------------------------------------------

def get_ibm_noise_params(n_qubits: int):
    """Generate per-qubit noise parameters based on IBM Nairobi specs."""
    specs = {
        't1_mean': 169e-6, 't1_std': 50e-6,
        't2_mean': 104e-6, 't2_std': 30e-6,
        'readout_error_mean': 0.0165, 'readout_error_std': 0.005,
        'gate_error_1q': 0.0003,
        'gate_error_2q': 0.0065,
    }

    np_rng = np.random.RandomState(42)  # deterministic noise params
    noise_params = {}

    for i in range(n_qubits):
        t1 = max(np_rng.normal(specs['t1_mean'], specs['t1_std']), 50e-6)
        t2 = max(np_rng.normal(specs['t2_mean'], specs['t2_std']), 30e-6)
        t2 = min(t2, t1 * 0.8)
        readout_error = np.clip(
            np_rng.normal(specs['readout_error_mean'], specs['readout_error_std']),
            0.005, 0.1
        )

        noise_params[f'qubit_{i}'] = {
            't1': t1,
            't2': t2,
            'amp_damping_1q': specs['gate_error_1q'] * 0.3,
            'amp_damping_2q': specs['gate_error_2q'] * 0.3,
            'phase_damping_1q': specs['gate_error_1q'] * 0.2,
            'phase_damping_2q': specs['gate_error_2q'] * 0.2,
            'depolarizing_1q': specs['gate_error_1q'] * 0.5,
            'depolarizing_2q': specs['gate_error_2q'] * 0.5,
            'readout_error': readout_error,
        }

    noise_params['readout_errors'] = [
        noise_params[f'qubit_{i}']['readout_error'] for i in range(n_qubits)
    ]
    return noise_params


# ---------------------------------------------------------------------------
# Configurable noisy quantum circuit
# ---------------------------------------------------------------------------

def create_noise_circuit(n_qubits, quantum_depth, noise_config):
    """
    Create a PennyLane QNode with only the specified noise components.

    noise_config: one of NOISE_CONFIGS
        - "ideal"             -> default.qubit, no noise channels
        - "amplitude_only"    -> default.mixed, AmplitudeDamping only
        - "phase_only"        -> default.mixed, PhaseDamping only
        - "depolarizing_only" -> default.mixed, DepolarizingChannel only
        - "full_noise"        -> default.mixed, all noise channels
    """
    noise_params = get_ibm_noise_params(n_qubits) if noise_config != "ideal" else {}

    use_amplitude = noise_config in ("amplitude_only", "full_noise")
    use_phase = noise_config in ("phase_only", "full_noise")
    use_depolarizing = noise_config in ("depolarizing_only", "full_noise")

    if noise_config == "ideal":
        dev = qml.device('default.qubit', wires=n_qubits)
    else:
        dev = qml.device('default.mixed', wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # Encode classical inputs
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)

            # Post-encoding noise
            if noise_config != "ideal":
                params = noise_params[f'qubit_{i}']
                if use_amplitude:
                    qml.AmplitudeDamping(params['amp_damping_1q'], wires=i)
                if use_phase:
                    qml.PhaseDamping(params['phase_damping_1q'], wires=i)
                if use_depolarizing:
                    qml.DepolarizingChannel(params['depolarizing_1q'], wires=i)

        # Variational layers with noise
        for layer in range(quantum_depth):
            # Entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

                if noise_config != "ideal":
                    for qubit in [i, i + 1]:
                        params = noise_params[f'qubit_{qubit}']
                        if use_amplitude:
                            qml.AmplitudeDamping(params['amp_damping_2q'], wires=qubit)
                        if use_phase:
                            qml.PhaseDamping(params['phase_damping_2q'], wires=qubit)
                        if use_depolarizing:
                            qml.DepolarizingChannel(params['depolarizing_2q'], wires=qubit)

            # Parameterized rotation layer
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)

                if noise_config != "ideal":
                    params = noise_params[f'qubit_{i}']
                    if use_amplitude:
                        qml.AmplitudeDamping(params['amp_damping_1q'], wires=i)
                    if use_phase:
                        qml.PhaseDamping(params['phase_damping_1q'], wires=i)
                    if use_depolarizing:
                        qml.DepolarizingChannel(params['depolarizing_1q'], wires=i)

        return qml.expval(qml.PauliZ(wires=0))

    return circuit


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------

class ConfigurableNoisyQuantumLayer(nn.Module):
    """Quantum layer with configurable noise components."""

    def __init__(self, n_qubits, quantum_depth, noise_config):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum_depth = quantum_depth
        self.noise_config = noise_config
        self.quantum_circuit = create_noise_circuit(n_qubits, quantum_depth, noise_config)
        self.weights = nn.Parameter(torch.randn(quantum_depth, n_qubits, 2) * 0.1)

    def forward(self, x):
        batch_size = x.size(0)
        results = []
        for i in range(batch_size):
            result = self.quantum_circuit(x[i], self.weights)
            result_tensor = torch.tensor(result, dtype=torch.float32)
            results.append(result_tensor)
        return torch.stack(results).unsqueeze(1)


class HybridNoiseDecompModel(nn.Module):
    """Hybrid classical-quantum model for noise decomposition experiments."""

    def __init__(self, backbone, feature_dim, n_qubits, num_classes, quantum_depth, noise_config):
        super().__init__()
        self.backbone = backbone
        self.pre_quantum = nn.Linear(feature_dim, n_qubits)
        self.quantum_layer = ConfigurableNoisyQuantumLayer(n_qubits, quantum_depth, noise_config)
        self.post_quantum = nn.Linear(1, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        quantum_input = torch.tanh(self.pre_quantum(features)) * np.pi
        quantum_output = self.quantum_layer(quantum_input)
        if quantum_output.dim() == 1:
            quantum_output = quantum_output.unsqueeze(1)
        return self.post_quantum(quantum_output)


# ---------------------------------------------------------------------------
# Index encoding / decoding
# ---------------------------------------------------------------------------

def decode_task_id(task_id: int):
    """Decode a linear task index into (noise_config, seed).

    Ordering (innermost to outermost): seed -> noise_config
    """
    n_seeds = len(SEEDS)
    if task_id < 0 or task_id >= TOTAL_EXPERIMENTS:
        raise ValueError(f"task-id {task_id} out of range [0, {TOTAL_EXPERIMENTS})")
    seed_idx = task_id % n_seeds
    config_idx = task_id // n_seeds
    return NOISE_CONFIGS[config_idx], SEEDS[seed_idx]


def make_run_id(noise_config: str, seed: int) -> str:
    return f"noise_{noise_config}_{BACKBONE}_{DATASET}_seed{seed}"


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def is_completed(output_dir: str, run_id: str) -> bool:
    path = os.path.join(output_dir, f"{run_id}.csv")
    if not os.path.isfile(path):
        return False
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_experiment(
    noise_config: str,
    seed: int,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    patience: int = 10,
):
    """Train and evaluate a single noise configuration with one seed."""

    run_id = make_run_id(noise_config, seed)
    print("=" * 70)
    print(f"NOISE DECOMPOSITION EXPERIMENT")
    print(f"  Config : {noise_config}")
    print(f"  Backbone: {BACKBONE} | Dataset: {DATASET}")
    print(f"  Qubits : {N_QUBITS} | Depth: {DEPTH}")
    print(f"  Seed   : {seed}")
    print(f"  Epochs : {epochs}")
    print(f"  Run ID : {run_id}")
    print("=" * 70)

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("Step 1/6: Loading dataset...")
    data_dir = _resolve_dataset_dir(DATASET)
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
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}, Classes: {num_classes}")

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------
    print("Step 2/6: Loading frozen backbone...")
    backbone = models.resnet18(pretrained=True)
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print(f"Step 3/6: Building hybrid model (noise_config={noise_config})...")
    model = HybridNoiseDecompModel(
        backbone, feat_dim, N_QUBITS, num_classes, DEPTH, noise_config,
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable_params}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("Step 4/6: Training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate,
    )

    # CodeCarbon
    energy_kwh = 0.0
    tracker = None
    if HAS_CODECARBON:
        try:
            tracker = EmissionsTracker(
                project_name=f"noise_decomp_{run_id}",
                log_level="error",
                save_to_file=False,
            )
            tracker.start()
        except Exception:
            tracker = None

    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, val_accs = [], [], []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validate
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

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start_time

    if tracker is not None:
        try:
            emissions = tracker.stop()
            if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data:
                energy_kwh = float(tracker.final_emissions_data.energy_consumed)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("Step 5/6: Evaluating on test set...")
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
    # Save CSV
    # ------------------------------------------------------------------
    print("Step 6/6: Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{run_id}.csv")
    header = (
        "noise_config,backbone,dataset,seed,n_qubits,quantum_depth,"
        "test_accuracy,precision,recall,f1,auc_roc,"
        "train_time_s,energy_kwh,"
        "epochs_actual,trainable_params,"
        "loss_history,val_acc_history"
    )
    row = (
        f"{noise_config},{BACKBONE},{DATASET},{seed},{N_QUBITS},{DEPTH},"
        f"{test_acc:.6f},{prec_w:.6f},{rec_w:.6f},{f1_w:.6f},{auc_roc_w:.6f},"
        f"{train_time:.2f},{energy_kwh:.8f},"
        f"{len(train_losses)},{trainable_params},"
        f"\"{json.dumps([round(v, 6) for v in train_losses])}\","
        f"\"{json.dumps([round(v, 6) for v in val_accs])}\""
    )
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(header + "\n")
        f.write(row + "\n")
    print(f"  CSV saved: {csv_path}")

    print("=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print(f"  Noise Config  : {noise_config}")
    print(f"  Seed          : {seed}")
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"  Train Time    : {train_time:.2f}s")
    print("=" * 70)

    return {
        'test_accuracy': test_acc,
        'precision': prec_w,
        'recall': rec_w,
        'f1': f1_w,
        'auc_roc': auc_roc_w,
        'train_time': train_time,
        'energy_kwh': energy_kwh,
    }


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def print_grid_summary():
    print("=" * 70)
    print("NOISE DECOMPOSITION EXPERIMENT GRID")
    print("=" * 70)
    print(f"  Noise configs ({len(NOISE_CONFIGS)}): {', '.join(NOISE_CONFIGS)}")
    print(f"  Seeds         ({len(SEEDS)}): {', '.join(str(s) for s in SEEDS)}")
    print(f"  Backbone: {BACKBONE}  |  Dataset: {DATASET}")
    print(f"  Qubits: {N_QUBITS}  |  Depth: {DEPTH}")
    print(f"  TOTAL RUNS: {TOTAL_EXPERIMENTS}")
    print("=" * 70)


def print_progress(done, total, successes, failures, skipped):
    pct = done / total * 100 if total else 0
    bar_len = 40
    filled = int(bar_len * done / total) if total else 0
    bar = "#" * filled + "-" * (bar_len - filled)
    print(
        f"\r  [{bar}] {done}/{total} ({pct:.1f}%) "
        f"| ok={successes} fail={failures} skip={skipped}",
        end="", flush=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Structured noise decomposition analysis. "
            f"Runs {len(NOISE_CONFIGS)} noise configs x {len(SEEDS)} seeds = "
            f"{TOTAL_EXPERIMENTS} experiments."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--task-id", type=int, default=None,
        help=f"Linear task index (0..{TOTAL_EXPERIMENTS - 1}) for SLURM array jobs.",
    )
    mode.add_argument(
        "--run-all", action="store_true", default=False,
        help=f"Run all {TOTAL_EXPERIMENTS} experiments sequentially.",
    )

    p.add_argument(
        "--resume", action="store_true", default=False,
        help="Skip experiments whose result CSV already exists.",
    )
    p.add_argument(
        "--output-dir", type=str, default=os.path.join("results", "noise"),
        help="Directory for per-run result CSVs (default: results/noise).",
    )
    p.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10).",
    )
    p.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16).",
    )
    p.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3).",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print what would run without executing.",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Single task mode (SLURM)
    # ------------------------------------------------------------------
    if args.task_id is not None:
        noise_config, seed = decode_task_id(args.task_id)
        run_id = make_run_id(noise_config, seed)

        print(f"Task ID {args.task_id} -> {run_id}")
        print(f"  noise_config={noise_config}  seed={seed}")

        if args.dry_run:
            print(f"[DRY-RUN] Would run: {run_id}")
            return 0

        if args.resume and is_completed(args.output_dir, run_id):
            print(f"SKIP (completed): {run_id}")
            return 0

        try:
            run_single_experiment(
                noise_config=noise_config,
                seed=seed,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )
            return 0
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # ------------------------------------------------------------------
    # Run-all mode
    # ------------------------------------------------------------------
    print_grid_summary()

    if args.dry_run:
        print("\n--- DRY RUN ---\n")
        for idx in range(TOTAL_EXPERIMENTS):
            nc, s = decode_task_id(idx)
            print(f"  [{idx:2d}] {make_run_id(nc, s)}")
        return 0

    successes = 0
    failures = 0
    skipped = 0
    t_start = time.time()

    for idx in range(TOTAL_EXPERIMENTS):
        noise_config, seed = decode_task_id(idx)
        run_id = make_run_id(noise_config, seed)

        if args.resume and is_completed(args.output_dir, run_id):
            skipped += 1
            print_progress(idx + 1, TOTAL_EXPERIMENTS, successes, failures, skipped)
            continue

        try:
            run_single_experiment(
                noise_config=noise_config,
                seed=seed,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )
            successes += 1
        except Exception as exc:
            print(f"\nFAIL {run_id}: {exc}")
            failures += 1

        print_progress(idx + 1, TOTAL_EXPERIMENTS, successes, failures, skipped)

    elapsed = time.time() - t_start
    print()  # newline after progress bar
    print("=" * 70)
    print("ALL NOISE DECOMPOSITION EXPERIMENTS FINISHED")
    print(f"  Total   : {TOTAL_EXPERIMENTS}")
    print(f"  Success : {successes}")
    print(f"  Failed  : {failures}")
    print(f"  Skipped : {skipped}")
    print(f"  Elapsed : {elapsed:.1f}s ({elapsed / 3600:.2f}h)")
    print("=" * 70)

    return 1 if failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
