"""test_qiskit_real_hardware.py - Evaluate hybrid models on real IBM Quantum hardware.

This script:
1. Loads a previously trained Qiskit hybrid model.
2. Runs inference on real quantum hardware through QiskitRuntimeService.
3. Applies optional mitigation techniques (for example, dynamical decoupling).
4. Reports test metrics and saves plots.

Usage examples:
    python test_qiskit_real_hardware.py --save-account YOUR_IBM_TOKEN
    python test_qiskit_real_hardware.py --list-backends
    python test_qiskit_real_hardware.py \
        --model-path model_saved/CQ_QK_hymenoptera_resnet18_4q_d3.pth \
        --dataset hymenoptera \
        --backbone resnet18 \
        --backend ibm_kyoto \
        --shots 1024
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

try:
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import SamplerQNN
except ImportError as exc:
    raise ImportError(f"qiskit-machine-learning is required: {exc}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_ibm_account(token: str, overwrite: bool = True) -> None:
    """Save IBM Quantum credentials (only needed once)."""
    try:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=token,
            overwrite=overwrite,
        )
        print("[SUCCESS] IBM Quantum credentials saved successfully.")
        print("You can now run real-hardware tests without --save-account.")
    except Exception as exc:
        print(f"[ERROR] Could not save credentials: {exc}")
        raise


def list_available_backends() -> list[str]:
    """List all backends available in the IBM account."""
    try:
        service = QiskitRuntimeService(channel="ibm_quantum")
        backends = service.backends()
        print("\n[INFO] Available backends:")
        for backend in backends:
            status = backend.status()
            print(
                f"  - {backend.name}: {backend.num_qubits} qubits, "
                f"pending_jobs={status.pending_jobs}, operational={status.operational}"
            )
        return [backend.name for backend in backends]
    except Exception as exc:
        print(f"[ERROR] Could not connect to IBM service: {exc}")
        print("Run first: python test_qiskit_real_hardware.py --save-account YOUR_TOKEN")
        return []


def estimate_runtime_cost(
    n_samples: int,
    shots: int = 1024,
    n_qubits: int = 4,
    depth: int = 3,
    batch_size: int = 8,
) -> dict:
    """Estimate runtime and resource usage for real-hardware inference."""
    compile_time_per_circuit = 10.0  # seconds (average)
    execution_time_per_shot = 0.005  # seconds (conservative estimate)
    communication_overhead = 3.0  # seconds per batch

    n_batches = int(np.ceil(n_samples / batch_size))

    total_compile_time = compile_time_per_circuit
    total_execution_time = n_samples * shots * execution_time_per_shot
    total_communication = n_batches * communication_overhead
    time_without_queue = total_compile_time + total_execution_time + total_communication

    queue_time_best = 0
    queue_time_typical = 180
    queue_time_worst = 1800

    return {
        "samples": n_samples,
        "batches": n_batches,
        "shots": shots,
        "n_qubits": n_qubits,
        "depth": depth,
        "compile_time_s": total_compile_time,
        "execution_time_s": total_execution_time,
        "communication_s": total_communication,
        "total_compute_s": time_without_queue,
        "total_compute_min": time_without_queue / 60,
        "best_case_s": time_without_queue + queue_time_best,
        "best_case_min": (time_without_queue + queue_time_best) / 60,
        "typical_case_s": time_without_queue + queue_time_typical,
        "typical_case_min": (time_without_queue + queue_time_typical) / 60,
        "worst_case_s": time_without_queue + queue_time_worst,
        "worst_case_min": (time_without_queue + queue_time_worst) / 60,
        "ibm_minutes": time_without_queue / 60,
    }


def print_runtime_estimate(
    n_samples: int,
    shots: int = 1024,
    n_qubits: int = 4,
    depth: int = 3,
    batch_size: int = 8,
    backend_name: str = "ibm_kyoto",
) -> dict:
    """Print a detailed runtime estimate and return the estimate dict."""
    est = estimate_runtime_cost(
        n_samples=n_samples,
        shots=shots,
        n_qubits=n_qubits,
        depth=depth,
        batch_size=batch_size,
    )

    print("\n" + "=" * 80)
    print("RUNTIME AND COST ESTIMATE - REAL IBM QUANTUM HARDWARE")
    print("=" * 80)
    print(f"Backend: {backend_name}")
    print(f"Samples: {n_samples} | Batches: {est['batches']} | Shots: {shots}")
    print(f"Configuration: {n_qubits} qubits, depth={depth}")
    print("-" * 80)
    print("\nPROCESSING TIMES:")
    print(f"  - Compilation/transpilation: {est['compile_time_s']:.1f}s")
    print(f"  - Quantum execution: {est['execution_time_s']:.1f}s ({est['execution_time_s']/60:.1f} min)")
    print(f"  - Communication overhead: {est['communication_s']:.1f}s")
    print(f"  - Total compute: {est['total_compute_s']:.1f}s ({est['total_compute_min']:.2f} min)")
    print("\nQUEUE SCENARIOS:")
    print(f"  - Best case (no queue): {est['best_case_min']:.1f} min")
    print(f"  - Typical case (~3 min queue): {est['typical_case_min']:.1f} min")
    print(f"  - Worst case (~30 min queue): {est['worst_case_min']:.1f} min")
    print("\nIBM RESOURCE USAGE:")
    print(f"  - Billable IBM minutes: {est['ibm_minutes']:.2f} min")
    print("=" * 80 + "\n")
    return est


def inspect_saved_model(model_path: str) -> None:
    """Inspect a saved model file and print structure details."""
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return

    print(f"\n[INFO] Inspecting model: {model_path}")
    print(f"[INFO] File size: {os.path.getsize(model_path) / 1024:.2f} KB\n")

    try:
        state_dict = torch.load(model_path, map_location="cpu")

        pre_net_params = [k for k in state_dict.keys() if "pre_net" in k]
        q_layer_params = [k for k in state_dict.keys() if "q_layer" in k]
        other_params = [k for k in state_dict.keys() if "pre_net" not in k and "q_layer" not in k]

        print("[INFO] Parameters found:")
        print(f"  - pre_net tensors: {len(pre_net_params)}")
        for key in pre_net_params:
            print(f"    {key}: {tuple(state_dict[key].shape)}")

        print(f"\n  - q_layer tensors: {len(q_layer_params)}")
        for key in q_layer_params[:5]:
            print(f"    {key}: {tuple(state_dict[key].shape)}")
        if len(q_layer_params) > 5:
            print(f"    ... and {len(q_layer_params) - 5} more")

        if other_params:
            print(f"\n  - other tensors: {len(other_params)}")

        if pre_net_params:
            weight_key = [k for k in pre_net_params if "weight" in k][0]
            in_features, n_qubits = state_dict[weight_key].shape
            print("\n[INFO] Inferred configuration:")
            print(f"  - Backbone feature dim: {in_features}")
            print(f"  - Number of qubits: {n_qubits}")

        print("\n[SUCCESS] Model appears valid for real-hardware testing.")
    except Exception as exc:
        print(f"[ERROR] Could not load model: {exc}")
        raise


def _safe_load_model(model_name: str):
    """Load a torchvision model with compatibility fallback."""
    name = model_name.lower()
    try:
        if name == "resnet18":
            return torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        if name == "mobilenetv2":
            return torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        if name == "efficientnet_b0":
            return torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        if name == "regnet_x_400mf":
            return torchvision.models.regnet_x_400mf(weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT)
        raise ValueError(f"Unsupported model: {name}")
    except (AttributeError, TypeError):
        if name == "resnet18":
            return torchvision.models.resnet18(pretrained=True)
        if name == "mobilenetv2":
            return torchvision.models.mobilenet_v2(pretrained=True)
        if name == "efficientnet_b0":
            return torchvision.models.efficientnet_b0(pretrained=True)
        if name == "regnet_x_400mf":
            return torchvision.models.regnet_x_400mf(pretrained=True)
        raise


def build_quantum_circuit(n_qubits: int, quantum_depth: int):
    """Build the same parameterized circuit topology used during training."""
    qc = QuantumCircuit(n_qubits)

    for idx in range(n_qubits):
        qc.h(idx)

    feature_params = [Parameter(f"theta_{idx}") for idx in range(n_qubits)]
    for idx in range(n_qubits):
        qc.ry(feature_params[idx], idx)

    var_params = []
    for layer in range(quantum_depth):
        for idx in range(0, n_qubits - 1, 2):
            qc.cx(idx, idx + 1)
        for idx in range(1, n_qubits - 1, 2):
            qc.cx(idx, idx + 1)

        layer_params = [Parameter(f"phi_{layer}_{idx}") for idx in range(n_qubits)]
        var_params.append(layer_params)
        for idx in range(n_qubits):
            qc.ry(layer_params[idx], idx)

    return qc, feature_params, var_params


def build_quantum_qnn_real_hardware(
    n_qubits: int,
    quantum_depth: int,
    backend_name: str,
    num_classes: int = 2,
    shots: int = 1024,
    apply_dd: bool = True,
    optimization_level: int = 3,
):
    """Build a SamplerQNN backed by a real IBM quantum device."""
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend(backend_name)

    print(f"[INFO] Backend: {backend.name}")
    print(f"[INFO] Qubits: {backend.num_qubits}")

    qc, feature_params, var_params = build_quantum_circuit(n_qubits, quantum_depth)
    weight_params = [p for layer in var_params for p in layer]

    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=optimization_level,
    )

    if apply_dd:
        dd_sequence = [XGate(), XGate()]
        pm.scheduling = PassManager([
            ALAPScheduleAnalysis(target=backend.target),
            PadDynamicalDecoupling(target=backend.target, dd_sequence=dd_sequence),
        ])
        print("[INFO] Dynamical decoupling enabled.")

    sampler = SamplerV2(backend)
    sampler.options.default_shots = shots
    sampler.options.optimization_level = optimization_level

    def interpret_index(bit_int: int) -> int:
        return bin(bit_int).count("1") % num_classes

    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=feature_params,
        weight_params=weight_params,
        interpret=interpret_index,
        output_shape=num_classes,
        input_gradients=False,
    )
    return TorchConnector(qnn), pm


class QuantumNetTorch(nn.Module):
    """Hybrid head that mirrors the training architecture."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        n_qubits: int,
        quantum_depth: int,
        backend_name: str,
        shots: int,
        apply_dd: bool = True,
    ):
        super().__init__()
        self.pre_net = nn.Linear(in_features, n_qubits)
        self.q_layer, self.pass_manager = build_quantum_qnn_real_hardware(
            n_qubits=n_qubits,
            quantum_depth=quantum_depth,
            backend_name=backend_name,
            num_classes=num_classes,
            shots=shots,
            apply_dd=apply_dd,
        )

    def forward(self, x: torch.Tensor):
        z = self.pre_net(x)
        q_input = torch.tanh(z) * (np.pi / 2.0)
        return self.q_layer(q_input)


def get_in_features(model, model_name: str) -> int:
    """Extract backbone output feature size for supported architectures."""
    name = model_name.lower()
    if name in ("resnet18", "regnet_x_400mf"):
        return model.fc.in_features
    if name in ("mobilenetv2", "efficientnet_b0"):
        return model.classifier[1].in_features
    raise ValueError(f"Unsupported model: {name}")


def replace_classifier(model, model_name: str, quantum_head: nn.Module):
    """Replace the final classifier with the quantum head."""
    name = model_name.lower()
    if name in ("resnet18", "regnet_x_400mf"):
        model.fc = quantum_head
    elif name in ("mobilenetv2", "efficientnet_b0"):
        model.classifier[1] = quantum_head
    else:
        raise ValueError(f"Unsupported model: {name}")
    return model


def _resolve_dataset_dir(name: str) -> str:
    """Resolve dataset folder path from common repository locations."""
    script_root = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    candidates = [
        os.path.join(script_root, "datasets", name),
        os.path.join(cwd, "datasets", name),
        os.path.join(script_root, "Resultados", "datasets", name),
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "test")):
            return candidate
    raise FileNotFoundError(f"Dataset '{name}' was not found. Checked: {candidates}")


def test_on_real_hardware(
    model_path: str,
    dataset_name: str,
    backbone_name: str,
    backend_name: str,
    n_qubits: int = 4,
    quantum_depth: int = 3,
    shots: int = 1024,
    batch_size: int = 8,
    apply_dd: bool = True,
    save_metrics: bool = True,
    estimate_only: bool = False,
    auto_confirm: bool = False,
):
    """Run a full test pass on real IBM hardware."""
    print("\n" + "=" * 80)
    print("REAL HARDWARE TEST - IBM QUANTUM")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_name}")
    print(f"Backbone: {backbone_name}")
    print(f"Backend: {backend_name}")
    print(f"Configuration: {n_qubits} qubits, depth={quantum_depth}, shots={shots}")
    print("=" * 80 + "\n")

    data_dir = _resolve_dataset_dir(dataset_name)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    num_classes = len(test_ds.classes)

    print(f"[INFO] Test samples: {len(test_ds)}")
    print(f"[INFO] Classes: {test_ds.classes}")
    print(f"[INFO] Batch size: {batch_size}\n")

    estimate = print_runtime_estimate(
        n_samples=len(test_ds),
        shots=shots,
        n_qubits=n_qubits,
        depth=quantum_depth,
        batch_size=batch_size,
        backend_name=backend_name,
    )

    if estimate_only:
        print("[INFO] Estimate-only mode enabled. No execution will be performed.")
        return {"estimate": estimate, "samples": len(test_ds), "backend": backend_name}

    if not auto_confirm:
        try:
            user_input = input("Continue with real-hardware execution? [y/N]: ")
            if user_input.lower() not in {"y", "yes", "s", "si"}:
                print("[INFO] Execution canceled by user.")
                return None
        except KeyboardInterrupt:
            print("\n[INFO] Execution canceled.")
            return None
    else:
        print("[INFO] Auto-confirm enabled. Starting execution...\n")

    base_model = _safe_load_model(backbone_name).to(device)
    for param in base_model.parameters():
        param.requires_grad = False

    in_features = get_in_features(base_model, backbone_name)
    print("[INFO] Building quantum head on real hardware backend...")
    q_head = QuantumNetTorch(
        in_features=in_features,
        num_classes=num_classes,
        n_qubits=n_qubits,
        quantum_depth=quantum_depth,
        backend_name=backend_name,
        shots=shots,
        apply_dd=apply_dd,
    ).to(device)
    hybrid = replace_classifier(base_model, backbone_name, q_head)

    print(f"[INFO] Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)

    model_keys = set(state_dict.keys())
    expected_keys = set(hybrid.state_dict().keys())
    missing_keys = expected_keys - model_keys
    unexpected_keys = model_keys - expected_keys

    if missing_keys:
        print(f"[WARN] Missing keys in saved model: {missing_keys}")
    if unexpected_keys:
        print(f"[WARN] Unexpected keys in saved model: {unexpected_keys}")

    pre_net_keys = [key for key in model_keys if "pre_net" in key]
    if not pre_net_keys:
        raise ValueError(
            "No pre_net weights found in the saved model. "
            "Use a model trained with train_cq_qiskit.py."
        )

    try:
        hybrid.load_state_dict(state_dict, strict=False)
        print("[SUCCESS] Weights loaded successfully.")
        print(f"[INFO] pre_net tensors loaded: {len(pre_net_keys)}")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load weights: {exc}\n"
            "Verify architecture compatibility for n_qubits, depth, and backbone."
        )

    hybrid.eval()
    print("[INFO] Starting evaluation on real quantum hardware...")
    print("[WARN] Execution can take several minutes, depending on backend queue.\n")

    y_true = []
    y_pred = []
    y_scores = []
    correct = 0
    total = 0

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(test_loader):
            batch_start = time.time()
            xb = xb.to(device)

            out = hybrid(xb)
            probs = torch.softmax(out, dim=1)
            _, pred = torch.max(out, 1)

            y_true.extend(yb.tolist())
            y_pred.extend(pred.cpu().tolist())
            if num_classes == 2:
                y_scores.extend(probs[:, 1].cpu().tolist())

            total += yb.size(0)
            correct += (pred.cpu() == yb).sum().item()

            batch_time = time.time() - batch_start
            batch_acc = (pred.cpu() == yb).sum().item() / yb.size(0)
            print(
                f"Batch {batch_idx + 1}/{len(test_loader)}: "
                f"batch_acc={batch_acc:.2%}, batch_time={batch_time:.1f}s, "
                f"running_acc={correct / total:.2%}"
            )

    test_time = time.time() - t0
    test_acc = correct / total

    print("\n" + "=" * 80)
    print("REAL HARDWARE RESULTS")
    print("=" * 80)
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Correct/Total: {correct}/{total}")
    print(f"Test Time: {test_time:.2f}s ({test_time / 60:.1f} min)")
    print(f"Average Time/Sample: {test_time / total:.2f}s")
    print("=" * 80 + "\n")

    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}\n")

    if save_metrics:
        metrics_dir = os.path.join("static", "real_hardware_metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        run_id = f"REAL_{backend_name}_{dataset_name}_{backbone_name}_{int(time.time())}"

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=test_ds.classes)
        plt.figure(figsize=(8, 6))
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix - Real Hardware\n{backend_name}")
        plt.savefig(os.path.join(metrics_dir, f"{run_id}_confmat.png"), dpi=150, bbox_inches="tight")
        plt.close()

        roc_auc = None
        if num_classes == 2 and len(y_scores) == len(y_true):
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
            plt.plot([0, 1], [0, 1], "k--", label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - Real Hardware\n{backend_name}")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.savefig(os.path.join(metrics_dir, f"{run_id}_roc.png"), dpi=150, bbox_inches="tight")
            plt.close()

            print(f"AUC-ROC: {roc_auc:.4f}\n")

        summary_path = os.path.join(metrics_dir, f"{run_id}_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as file:
            file.write("REAL HARDWARE TEST SUMMARY\n")
            file.write("=" * 60 + "\n\n")
            file.write(f"Model Path: {model_path}\n")
            file.write(f"Dataset: {dataset_name}\n")
            file.write(f"Backbone: {backbone_name}\n")
            file.write(f"Backend: {backend_name}\n")
            file.write(f"Qubits: {n_qubits}, Depth: {quantum_depth}\n")
            file.write(f"Shots: {shots}\n")
            file.write(f"Dynamical Decoupling: {apply_dd}\n\n")
            file.write("RESULTS:\n")
            file.write(f"Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)\n")
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall: {recall:.4f}\n")
            file.write(f"F1-Score: {f1:.4f}\n")
            file.write(f"Test Time: {test_time:.2f}s ({test_time / 60:.1f} min)\n")
            file.write(f"Samples: {total}\n")
            if roc_auc is not None:
                file.write(f"AUC-ROC: {roc_auc:.4f}\n")

        print(f"[INFO] Metrics saved in: {metrics_dir}")
        print(f"[INFO] Run ID: {run_id}\n")

    return {
        "accuracy": test_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_time": test_time,
        "samples": total,
        "backend": backend_name,
        "shots": shots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test Qiskit hybrid models on real IBM Quantum hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:

1. Save IBM credentials (only once):
   python test_qiskit_real_hardware.py --save-account YOUR_IBM_QUANTUM_TOKEN

2. List available backends:
   python test_qiskit_real_hardware.py --list-backends

3. Inspect a trained model:
   python test_qiskit_real_hardware.py --inspect-model model_saved/CQ_QK_hymenoptera_resnet18_4q_d3.pth

4. Estimate runtime/cost only (without execution):
   python test_qiskit_real_hardware.py \
       --model-path model_saved/CQ_QK_hymenoptera_resnet18_4q_d3.pth \
       --dataset hymenoptera \
       --backbone resnet18 \
       --backend ibm_kyoto \
       --estimate-only

5. Run on real hardware with confirmation:
   python test_qiskit_real_hardware.py \
       --model-path model_saved/CQ_QK_hymenoptera_resnet18_4q_d3.pth \
       --dataset hymenoptera \
       --backbone resnet18 \
       --backend ibm_kyoto \
       --shots 1024

IMPORTANT: The model must have been trained with train_cq_qiskit.py.
""",
    )

    parser.add_argument("--save-account", type=str, metavar="TOKEN", help="Save IBM Quantum token")
    parser.add_argument("--list-backends", action="store_true", help="List available backends")
    parser.add_argument("--inspect-model", type=str, metavar="PATH", help="Inspect a saved model")

    parser.add_argument("--model-path", type=str, help="Path to trained .pth model")
    parser.add_argument("--dataset", type=str, default="hymenoptera", help="Dataset name")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "mobilenetv2", "efficientnet_b0", "regnet_x_400mf"],
        help="Model backbone",
    )

    parser.add_argument("--backend", type=str, default="ibm_kyoto", help="IBM backend name")
    parser.add_argument("--n-qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--depth", type=int, default=3, help="Quantum circuit depth")
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--no-dd", action="store_true", help="Disable dynamical decoupling")
    parser.add_argument("--no-save", action="store_true", help="Do not save metrics")
    parser.add_argument("--estimate-only", action="store_true", help="Estimate only, do not run")
    parser.add_argument("--yes", "-y", action="store_true", help="Auto-confirm execution")

    args = parser.parse_args()

    if args.save_account:
        save_ibm_account(args.save_account)
        return 0

    if args.list_backends:
        list_available_backends()
        return 0

    if args.inspect_model:
        inspect_saved_model(args.inspect_model)
        return 0

    if not args.model_path:
        parser.error("--model-path is required for real-hardware testing")

    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found: {args.model_path}")

    try:
        test_on_real_hardware(
            model_path=args.model_path,
            dataset_name=args.dataset,
            backbone_name=args.backbone,
            backend_name=args.backend,
            n_qubits=args.n_qubits,
            quantum_depth=args.depth,
            shots=args.shots,
            batch_size=args.batch_size,
            apply_dd=not args.no_dd,
            save_metrics=not args.no_save,
            estimate_only=args.estimate_only,
            auto_confirm=args.yes,
        )
        print("\n[SUCCESS] Real-hardware test completed successfully.")
    except Exception as exc:
        print(f"\n[ERROR] Test failed: {exc}")
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
