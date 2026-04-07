"""train_cq_qiskit.py
Simplified implementation compatible with Qiskit >=1.x, avoiding EstimatorQNN and removed APIs.
Uses only SamplerQNN and param-shift gradients when available.
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import copy
import time
import os
import numpy as np
import argparse
from datetime import datetime

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
# Note: qiskit-aer imports are performed lazily
try:
    from qiskit_machine_learning.neural_networks import SamplerQNN  # type: ignore
    from qiskit_machine_learning.connectors import TorchConnector  # type: ignore
except Exception as _imp_err:  # pragma: no cover
    SamplerQNN = None  # type: ignore
    TorchConnector = None  # type: ignore
    _QISKIT_ML_IMPORT_ERROR = _imp_err
    if SamplerQNN is None or TorchConnector is None:
        raise ImportError(f"qiskit-machine-learning no disponible o incompatible: {_QISKIT_ML_IMPORT_ERROR}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _safe_load_model(model_name: str):
    """Load a torchvision model robustly across pretrained/weights API changes."""
    name = model_name.lower()
    try:
        if name == "resnet18":
            return torchvision.models.resnet18(weights=getattr(torchvision.models, 'ResNet18_Weights', None).DEFAULT if hasattr(torchvision.models, 'ResNet18_Weights') else None)
        if name == "resnet34":
            return torchvision.models.resnet34(weights=getattr(torchvision.models, 'ResNet34_Weights', None).DEFAULT if hasattr(torchvision.models, 'ResNet34_Weights') else None)
        if name == "vgg16":
            return torchvision.models.vgg16(weights=getattr(torchvision.models, 'VGG16_Weights', None).DEFAULT if hasattr(torchvision.models, 'VGG16_Weights') else None)
        if name == "vgg19":
            return torchvision.models.vgg19(weights=getattr(torchvision.models, 'VGG19_Weights', None).DEFAULT if hasattr(torchvision.models, 'VGG19_Weights') else None)
        if name == "mobilenetv2":
            return torchvision.models.mobilenet_v2(weights=getattr(torchvision.models, 'MobileNet_V2_Weights', None).DEFAULT if hasattr(torchvision.models, 'MobileNet_V2_Weights') else None)
        raise ValueError("Unsupported classical model")
    except TypeError:
        # Fallback legacy API
        if name == "resnet18":
            return torchvision.models.resnet18(pretrained=True)
        if name == "resnet34":
            return torchvision.models.resnet34(pretrained=True)
        if name == "vgg16":
            return torchvision.models.vgg16(pretrained=True)
        if name == "vgg19":
            return torchvision.models.vgg19(pretrained=True)
        if name == "mobilenetv2":
            return torchvision.models.mobilenet_v2(pretrained=True)
        raise

def build_quantum_circuit(n_qubits: int, quantum_depth: int):
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)
    feature_params = [Parameter(f"θ_{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        qc.ry(feature_params[i], i)
    var_params = []
    for layer in range(quantum_depth):
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        layer_params = [Parameter(f"φ_{layer}_{i}") for i in range(n_qubits)]
        var_params.append(layer_params)
        for i in range(n_qubits):
            qc.ry(layer_params[i], i)
    return qc, feature_params, var_params

def build_quantum_qnn(n_qubits: int, quantum_depth: int, num_classes: int = 2,
                      shots: int | None = None,
                      use_noise: bool = False,
                      noise_1q: float = 0.0,
                      noise_2q: float = 0.0):
    qc, feature_params, var_params = build_quantum_circuit(n_qubits, quantum_depth)
    weight_params = [p for sub in var_params for p in sub]

    def interpret_index(bit_int: int) -> int:
        # Parity mod num_classes (for 2 classes this provides a simple mapping)
        return bin(bit_int).count("1") % num_classes

    # Build sampler: AerSampler with optional noise model and shots, else default Sampler
    sampler: Sampler
    if shots is not None or use_noise:
        try:
            from qiskit_aer import AerSimulator  # type: ignore
            from qiskit_aer.primitives import Sampler as AerSampler  # type: ignore
            from qiskit_aer.noise import NoiseModel, depolarizing_error  # type: ignore
            
            # Configure backend with optional noise
            backend_options = {}
            if use_noise and (noise_1q > 0.0 or noise_2q > 0.0):
                noise_model = NoiseModel()
                if noise_1q > 0.0:
                    e1 = depolarizing_error(noise_1q, 1)
                    for g in ['x', 'y', 'z', 'rx', 'ry', 'rz', 'h', 'sx', 'id']:
                        try:
                            noise_model.add_all_qubit_quantum_error(e1, [g])
                        except Exception:
                            pass
                if noise_2q > 0.0:
                    e2 = depolarizing_error(noise_2q, 2)
                    for g2 in ['cx', 'cz', 'swap']:
                        try:
                            noise_model.add_all_qubit_quantum_error(e2, [g2])
                        except Exception:
                            pass
                backend_options['noise_model'] = noise_model
                print(f"[INFO] Noise configured: 1q={noise_1q}, 2q={noise_2q}")
            
            # Create AerSimulator backend
            backend = AerSimulator(**backend_options)
            
            # Create AerSampler with the correct API
            sampler = AerSampler()
            sampler.set_options(backend=backend)
            if shots is not None:
                sampler.set_options(shots=shots)
                print(f"[INFO] Shots configured: {shots}")
            else:
                sampler.set_options(shots=1024)
                
        except Exception as e:
            print(f"[WARN] qiskit-aer not available or incompatible: {e}")
            print("[INFO] Using basic Sampler without noise/shots")
            sampler = Sampler()
    else:
        sampler = Sampler()
    # Optional gradient (older versions may not include it)
    try:
        from qiskit_machine_learning.gradients import ParamShiftSamplerGradient  # type: ignore
        gradient = ParamShiftSamplerGradient(sampler=sampler)
    except Exception as e:  # pragma: no cover
        print("Param-shift gradient unavailable, continuing without explicit gradients:", e)
        gradient = None

    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=feature_params,
        weight_params=weight_params,
        interpret=interpret_index,
        output_shape=num_classes,
        gradient=gradient,
        input_gradients=True
    )
    return TorchConnector(qnn)

class QuantumNetTorch(nn.Module):
    """Hybrid layer: linear projection -> scaling -> QNN."""
    def __init__(self, in_features: int, num_classes: int, n_qubits: int, quantum_depth: int):
        super().__init__()
        self.pre_net = nn.Linear(in_features, n_qubits)
        self.q_layer = build_quantum_qnn(n_qubits, quantum_depth, num_classes=num_classes)

    def forward(self, x: torch.Tensor):  # type: ignore
        z = self.pre_net(x)
        q_input = torch.tanh(z) * (np.pi / 2.0)
        return self.q_layer(q_input)

def get_in_features(model, model_name: str):
    name = model_name.lower()
    if name in ("resnet18", "resnet34"):
        return model.fc.in_features
    if name == "vgg16":
        return model.classifier[6].in_features
    if name == "vgg19":
        return 25088  # flatten known constant
    if name == "mobilenetv2":
        return model.classifier[1].in_features
    raise ValueError("Unsupported model for quantum hybrid")

def replace_classifier(model, model_name: str, quantum_head: nn.Module):
    name = model_name.lower()
    if name in ("resnet18", "resnet34"):
        model.fc = quantum_head
    elif name == "vgg16":
        model.classifier[6] = quantum_head
    elif name == "vgg19":
        model.classifier = nn.Sequential(nn.Flatten(), quantum_head)
    elif name == "mobilenetv2":
        model.classifier[1] = quantum_head
    else:
        raise ValueError("Unsupported model for quantum hybrid")
    return model

def train_quantum_hybrid_qiskit(dataset_file="hymenoptera", classical_model="resnet18", n_qubits=4, quantum_depth=3,
                                epochs=20, id="null", batch_size=32, learning_rate=0.001, gamma=0.9,
                                shots: int | None = None, use_noise: bool = False,
                                noise_1q: float = 0.0, noise_2q: float = 0.0):
    print("============================================================")
    print("Qiskit Quantum Transfer Learning")
    print("============================================================")
    print(f"Dataset: {dataset_file}")
    print(f"Model: {classical_model}")
    print(f"Qubits: {n_qubits}")
    print(f"Depth: {quantum_depth}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"ID: {id}")
    print("============================================================")
    
    print("Step 1/7: Loading and preparing datasets...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    def _resolve_dataset_dir(name: str):
        script_root = os.path.dirname(os.path.abspath(__file__))
        cwd = os.getcwd()
        candidates = [
            os.path.join(script_root, 'Resultados', 'datasets', name),
            os.path.join(script_root, 'datasets', name),
            os.path.join(cwd, 'Resultados', 'datasets', name),
            os.path.join(cwd, 'datasets', name),
            os.path.join(script_root, 'user_datasets', name),
            os.path.join(cwd, 'user_datasets', name)
        ]
        for c in candidates:
            if os.path.isdir(os.path.join(c, 'train')) and os.path.isdir(os.path.join(c, 'test')):
                return c
        raise FileNotFoundError(f"Dataset '{name}' was not found. Checked paths: " + ' | '.join(candidates))
    data_dir = _resolve_dataset_dir(dataset_file)
    print(f"Using dataset path: {data_dir}")
    print(f"Train path: {os.path.join(data_dir, 'train')}")
    print(f"Test path: {os.path.join(data_dir, 'test')}")
    
    full_train = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=transform)
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    num_classes = len(full_train.classes)

    print("Step 2/7: Defining classical base model...")
    base_model = _safe_load_model(classical_model).to(device)
    for p in base_model.parameters():
        p.requires_grad = False
    in_features = get_in_features(base_model, classical_model)
    
    print("Step 3/7: Creating quantum circuit and hybrid model...")
    # Build quantum head with configured sampler/noise
    q_layer = build_quantum_qnn(n_qubits, quantum_depth, num_classes=num_classes,
                                shots=shots, use_noise=use_noise,
                                noise_1q=noise_1q, noise_2q=noise_2q)
    class QuantumNetTorch(nn.Module):
        def __init__(self, in_features: int):
            super().__init__()
            self.pre_net = nn.Linear(in_features, n_qubits)
            self.q_layer = q_layer
        def forward(self, x: torch.Tensor):
            z = self.pre_net(x)
            q_input = torch.tanh(z) * (np.pi / 2.0)
            return self.q_layer(q_input)
    q_head = QuantumNetTorch(in_features).to(device)
    hybrid = replace_classifier(base_model, classical_model, q_head)

    print("Step 4/7: Setting up training components...")
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam([p for p in hybrid.parameters() if p.requires_grad], lr=learning_rate)
    sched = lr_scheduler.StepLR(opt, step_size=10, gamma=gamma)

    def eval_loader(loader):
        hybrid.eval()
        correct = total = 0
        with torch.no_grad():
            for xi, yi in loader:
                xi, yi = xi.to(device), yi.to(device)
                out = hybrid(xi)
                _, pr = torch.max(out, 1)
                total += yi.size(0)
                correct += (pr == yi).sum().item()
        return correct / total if total else 0.0

    print("Step 5/7: Starting training...")
    losses, val_accs, val_losses = [], [], []
    best_w, best_acc, best_ep = copy.deepcopy(hybrid.state_dict()), 0.0, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        hybrid.train()
        run_loss = 0.0
        for xi, yi in train_loader:
            xi, yi = xi.to(device), yi.to(device)
            opt.zero_grad()
            out = hybrid(xi)
            loss = crit(out, yi)
            loss.backward()
            opt.step()
            run_loss += loss.item()
        sched.step()
        # Validation loop to compute val loss & acc
        hybrid.eval()
        v_run_loss = 0.0
        v_batches = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                vout = hybrid(vx)
                vloss = crit(vout, vy)
                v_run_loss += vloss.item()
                v_batches += 1
                _, vpred = torch.max(vout, 1)
                total += vy.size(0)
                correct += (vpred == vy).sum().item()
        vacc = correct / total if total else 0.0
        epoch_train_loss = run_loss / len(train_loader)
        epoch_val_loss = v_run_loss / v_batches if v_batches else 0.0
        losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accs.append(vacc)
        print(f"Epoch {ep}/{epochs} - train_loss={epoch_train_loss:.4f} val_loss={epoch_val_loss:.4f} val_acc={vacc:.4f}")
        if vacc > best_acc:
            best_acc, best_w, best_ep = vacc, copy.deepcopy(hybrid.state_dict()), ep
    train_time = time.time() - t0
    hybrid.load_state_dict(best_w)
    print(f"Best val_acc={best_acc:.2%} epoch={best_ep}")
    
    print("Step 6/7: Evaluating model on test set...")
    t_test0 = time.time()
    test_acc = eval_loader(test_loader)
    test_time = time.time() - t_test0
    print(f"Test acc={test_acc:.2%} test_time={test_time:.2f}s train_time={train_time:.2f}s")

    print("Step 7/7: Saving model and comprehensive metrics...")
    os.makedirs("model_saved", exist_ok=True)
    torch.save(hybrid.state_dict(), os.path.join("model_saved", f"CQ_{id}_{classical_model}_{dataset_file}.pth"))

    metrics_dir = os.path.join("static", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    plt.figure(); plt.plot(range(1, epochs + 1), losses, label='Train Loss'); plt.plot(range(1, epochs + 1), val_losses, label='Val Loss'); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss"); plt.legend(); plt.savefig(os.path.join(metrics_dir, f"{id}_loss.png")); plt.close()
    plt.figure(); plt.plot(range(1, epochs + 1), val_accs); plt.xlabel("Epoch"); plt.ylabel("Val Acc"); plt.title("Val Acc"); plt.savefig(os.path.join(metrics_dir, f"{id}_acc.png")); plt.close()

    y_true, y_pred = [], []
    hybrid.eval()
    with torch.no_grad():
        for xi, yi in test_loader:
            xi = xi.to(device)
            out = hybrid(xi)
            _, pr = torch.max(out, 1)
            y_true.extend(yi.tolist())
            y_pred.extend(pr.cpu().tolist())
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=full_train.classes)
    disp.plot(); plt.title("Confusion Matrix"); plt.savefig(os.path.join(metrics_dir, f"{id}_confmat.png")); plt.close()
    # Print summary lists
    print('Val Loss per epoch:', [float(f"{v:.4f}") for v in val_losses])
    print('Val Acc  per epoch:', [float(f"{v:.4f}") for v in val_accs])
    return test_acc, train_time, test_time

def _build_arg_parser():
    p = argparse.ArgumentParser(description="Train a Qiskit SamplerQNN hybrid model")
    p.add_argument('--dataset', default='hymenoptera')
    p.add_argument('--model', default='resnet18', choices=['resnet18','resnet34','vgg16','vgg19','mobilenetv2'])
    p.add_argument('--n-qubits', type=int, default=4)
    p.add_argument('--depth', type=int, default=3, help='Quantum depth (quantum_depth)')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.9)
    p.add_argument('--shots', type=int, default=None)
    p.add_argument('--noise', action='store_true', help='Enable depolarizing noise model')
    p.add_argument('--noise-1q', type=float, default=0.001)
    p.add_argument('--noise-2q', type=float, default=0.01)
    p.add_argument('--id', default=None, help='Run identifier (auto-generated if not provided)')
    return p

if __name__ == '__main__':
    parser = _build_arg_parser()
    args = parser.parse_args()
    run_id = args.id or f"QK_{args.dataset}_{args.model}_{args.n_qubits}q_d{args.depth}_{int(time.time())}"
    print('Parameters:', args)
    train_quantum_hybrid_qiskit(
        dataset_file=args.dataset,
        classical_model=args.model,
        n_qubits=args.n_qubits,
        quantum_depth=args.depth,
        epochs=args.epochs,
        id=run_id,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        shots=args.shots,
        use_noise=args.noise,
        noise_1q=args.noise_1q,
        noise_2q=args.noise_2q
    )
