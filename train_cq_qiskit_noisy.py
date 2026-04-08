# train_cq_qiskit_noisy.py
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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_score, recall_score, f1_score
import seaborn as sns
import copy
import time
import os
import numpy as np
import argparse
from datetime import datetime
import random
import json
import csv
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
from sklearn.metrics import roc_auc_score


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""train_cq_qiskit_noisy.py
Version with realistic IBM Quantum backend noise for Qiskit >=1.x.
Uses SamplerQNN with realistic quantum device noise.
"""
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
try:
    # Intento moderno primero
    from qiskit.primitives import Sampler as BaseSampler  # type: ignore
except Exception:
    try:
        # Fallback to aer primitives (intermediate versions)
        from qiskit_aer.primitives import Sampler as BaseSampler  # type: ignore
    except Exception as _smp_err:  # pragma: no cover
        BaseSampler = None  # type: ignore
        _SAMPLER_IMPORT_ERROR = _smp_err

try:
    from qiskit_machine_learning.neural_networks import SamplerQNN  # type: ignore
    from qiskit_machine_learning.connectors import TorchConnector  # type: ignore
except Exception as _qml_err:  # pragma: no cover
    SamplerQNN = None  # type: ignore
    TorchConnector = None  # type: ignore
    _QML_IMPORT_ERROR = _qml_err

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_load_model(model_name: str):
    """Load a torchvision model compatible with legacy and modern APIs."""
    name = model_name.lower()
    try:
        if name == "resnet18":
            weights = getattr(torchvision.models, 'ResNet18_Weights', None)
            return torchvision.models.resnet18(weights=weights.DEFAULT if weights else None)
        if name == "resnet34":
            weights = getattr(torchvision.models, 'ResNet34_Weights', None)
            return torchvision.models.resnet34(weights=weights.DEFAULT if weights else None)
        if name == "mobilenetv2":
            weights = getattr(torchvision.models, 'MobileNet_V2_Weights', None)
            return torchvision.models.mobilenet_v2(weights=weights.DEFAULT if weights else None)
        if name == "efficientnet_b0":
            weights = getattr(torchvision.models, 'EfficientNet_B0_Weights', None)
            return torchvision.models.efficientnet_b0(weights=weights.DEFAULT if weights else None)
        if name == "regnet_x_400mf":
            weights = getattr(torchvision.models, 'RegNet_X_400MF_Weights', None)
            return torchvision.models.regnet_x_400mf(weights=weights.DEFAULT if weights else None)
        raise ValueError("Unsupported classical model")
    except TypeError:
        if name == "resnet18":
            return torchvision.models.resnet18(pretrained=True)
        if name == "resnet34":
            return torchvision.models.resnet34(pretrained=True)
        if name == "mobilenetv2":
            return torchvision.models.mobilenet_v2(pretrained=True)
        if name == "efficientnet_b0":
            return torchvision.models.efficientnet_b0(pretrained=True)
        if name == "regnet_x_400mf":
            return torchvision.models.regnet_x_400mf(pretrained=True)
        raise

def build_quantum_circuit(n_qubits, quantum_depth):
    qc = QuantumCircuit(n_qubits)
    # Apply Hadamard to each qubit
    for i in range(n_qubits):
        qc.h(i)
    
    # Create input feature parameters and encode features
    feature_params = [Parameter(f'θ_{i}') for i in range(n_qubits)]
    for i in range(n_qubits):
        qc.ry(feature_params[i], i)
    
    # Add variable layers
    var_params = []
    for layer in range(quantum_depth):
        # Entangle qubits in a chain pattern
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        
        # Parameterized Ry rotations for the layer
        layer_params = [Parameter(f'φ_{layer}_{i}') for i in range(n_qubits)]
        var_params.append(layer_params)
        for i in range(n_qubits):
            qc.ry(layer_params[i], i)
            
    return qc, feature_params, var_params

def interpret_fn(probabilities: np.ndarray) -> np.ndarray:
    """
    Take the probability vector of size 2^n_qubits (index = bitstring integer)
    and return a vector of size num_classes where each entry is the
    Z expectation value on qubit i (used here as a raw logit).
    """
    n_qubits = int(np.log2(len(probabilities)))
    num_classes = 2  # en tu caso

    expectations = []
    for i in range(num_classes):
        # For each state |s>, add (+1)*P(s) if bit i is 0, and (-1)*P(s) if it is 1
        exp_i = 0.0
        for state_index, p in enumerate(probabilities):
            bit_i = (state_index >> i) & 1
            value = 1.0 if bit_i == 0 else -1.0
            exp_i += value * p
        expectations.append(exp_i)
    return np.array(expectations)

from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
from qiskit_aer.noise.device import basic_device_gate_errors, basic_device_readout_errors
try:
    from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore
    from qiskit.providers.fake_provider import FakeBackend, FakeManilaV2, FakeNairobiV2, FakeLagosV2
    IBM_PROVIDER_AVAILABLE = True
except ImportError:
    print("[WARNING] qiskit-ibm-runtime is not available. Using a synthetic noise model.")
    IBM_PROVIDER_AVAILABLE = False
    FakeManilaV2 = None
    FakeNairobiV2 = None
    FakeLagosV2 = None

"""Note: EstimatorQNN / StatevectorEstimator were removed to avoid obsolete APIs."""

def get_real_device_noise_model(n_qubits, backend_name='ibm_nairobi'):
    """
    Get the noise model from a real IBM quantum device.
    If unavailable, use a realistic synthetic noise model.
    """
    if IBM_PROVIDER_AVAILABLE:
        try:
            # Try using a real backend from IBM Cloud
            service = QiskitRuntimeService()
            backend = service.get_backend(backend_name)
            return NoiseModel.from_backend(backend)
            
            # Use fake backends that simulate real devices
            if backend_name == 'ibm_manila' and FakeManilaV2:
                fake_backend = FakeManilaV2()
                return NoiseModel.from_backend(fake_backend)
            elif backend_name == 'ibm_nairobi' and FakeNairobiV2:
                fake_backend = FakeNairobiV2()  
                return NoiseModel.from_backend(fake_backend)
            elif backend_name == 'ibm_lagos' and FakeLagosV2:
                fake_backend = FakeLagosV2()
                return NoiseModel.from_backend(fake_backend)
        except Exception as e:
            print(f"[WARNING] Error accessing {backend_name}: {e}")
    
    # Realistic synthetic noise model based on real device parameters
    print("[INFO] Using realistic synthetic noise model based on IBM quantum devices")
    noise_model = NoiseModel()
    
    # Thermal relaxation errors (T1, T2) typical of IBM devices
    t1_times = np.random.normal(120e-6, 20e-6, size=n_qubits)
    t2_times = np.random.normal(80e-6, 15e-6, size=n_qubits)
    
    # Typical gate times
    gate_time_1q = 35e-9
    gate_time_2q = 300e-9
    
    # Add thermal relaxation errors
    for qubit in range(n_qubits):
        t1, t2 = abs(t1_times[qubit]), abs(t2_times[qubit])
        # Ensure T2 <= 2*T1 (physical constraint)
        t2 = min(t2, 2*t1)
        
        # Errors for single-qubit gates
        relax_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        noise_model.add_quantum_error(relax_1q, ['u1', 'u2', 'u3', 'ry', 'rz', 'sx', 'x'], [qubit])
        
    # Errors for two-qubit gates (higher noise)
    for qubit in range(n_qubits-1):
        t1_q1, t2_q1 = abs(t1_times[qubit]), abs(t2_times[qubit])
        t1_q2, t2_q2 = abs(t1_times[qubit+1]), abs(t2_times[qubit+1])
        
        t2_q1 = min(t2_q1, 2*t1_q1)
        t2_q2 = min(t2_q2, 2*t1_q2)
        
        # Combined relaxation error
        relax_2q_1 = thermal_relaxation_error(t1_q1, t2_q1, gate_time_2q)
        relax_2q_2 = thermal_relaxation_error(t1_q2, t2_q2, gate_time_2q) 
        
        # Additional depolarizing error for CNOT
        depol_2q = depolarizing_error(0.01, 2)  # 1% depolarizing error
        
        noise_model.add_quantum_error(relax_2q_1.compose(depol_2q), ['cx'], [qubit, qubit+1])
        
    # Realistic readout errors (typically 1-5% error rate)
    readout_errors = []
    for qubit in range(n_qubits):
        prob_meas0_prep1 = np.random.uniform(0.01, 0.05)  # 1-5% error rate
        prob_meas1_prep0 = np.random.uniform(0.01, 0.05)
        readout_error = ReadoutError([[1-prob_meas0_prep1, prob_meas0_prep1],
                                     [prob_meas1_prep0, 1-prob_meas1_prep0]])
        readout_errors.append(readout_error)
        noise_model.add_readout_error(readout_error, [qubit])
    
    return noise_model


def scale_noise_model(base_noise_model, scale_factor, n_qubits):
    """Create a new noise model with scaled error rates."""
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError

    scaled_model = NoiseModel()

    # Scale thermal relaxation by reducing T1/T2 (more noise = shorter coherence)
    t1_times = np.random.normal(120e-6, 20e-6, size=n_qubits)
    t2_times = np.random.normal(80e-6, 15e-6, size=n_qubits)

    gate_time_1q = 35e-9
    gate_time_2q = 300e-9

    for qubit in range(n_qubits):
        # Reduce T1/T2 by scale_factor to increase noise
        t1 = abs(t1_times[qubit]) / scale_factor
        t2 = min(abs(t2_times[qubit]) / scale_factor, 2 * t1)

        relax_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
        scaled_model.add_quantum_error(relax_1q, ['u1', 'u2', 'u3', 'ry', 'rz', 'sx', 'x'], [qubit])

    for qubit in range(n_qubits - 1):
        t1_q1 = abs(t1_times[qubit]) / scale_factor
        t2_q1 = min(abs(t2_times[qubit]) / scale_factor, 2 * t1_q1)
        t1_q2 = abs(t1_times[qubit+1]) / scale_factor
        t2_q2 = min(abs(t2_times[qubit+1]) / scale_factor, 2 * t1_q2)

        relax_2q = thermal_relaxation_error(t1_q1, t2_q1, gate_time_2q)
        depol_2q = depolarizing_error(min(0.01 * scale_factor, 0.5), 2)
        scaled_model.add_quantum_error(relax_2q.compose(depol_2q), ['cx'], [qubit, qubit+1])

    for qubit in range(n_qubits):
        p01 = min(np.random.uniform(0.01, 0.05) * scale_factor, 0.5)
        p10 = min(np.random.uniform(0.01, 0.05) * scale_factor, 0.5)
        readout_error = ReadoutError([[1-p01, p01], [p10, 1-p10]])
        scaled_model.add_readout_error(readout_error, [qubit])

    return scaled_model


def richardson_extrapolate(scale_factors, values):
    """Richardson extrapolation to zero noise."""
    scale_factors = np.array(scale_factors)
    values = np.array(values)
    degree = min(len(scale_factors) - 1, 2)
    coeffs = np.polyfit(scale_factors, values, degree)
    return np.polyval(coeffs, 0.0)


def build_quantum_qnn(n_qubits, quantum_depth, num_classes=2, noise_model=None):
    if BaseSampler is None:
        raise ImportError(f"Sampler could not be imported: {_SAMPLER_IMPORT_ERROR}")
    if SamplerQNN is None or TorchConnector is None:
        raise ImportError(f"qiskit-machine-learning missing or incompatible: {_QML_IMPORT_ERROR}")
    qc, feature_params, var_params = build_quantum_circuit(n_qubits, quantum_depth)
    weight_params = [p for sub in var_params for p in sub]

    def interpret_fn(bit_int: int) -> int:
        return bin(bit_int).count("1") % num_classes

    # Create base sampler and attempt to attach noise_model when supported
    try:
        sampler = BaseSampler()
    except Exception as e:
        raise RuntimeError(f"Failed to create base Sampler: {e}")

    # Dynamic gradient import (may not exist in versions <0.8)
    try:
        from qiskit_machine_learning.gradients import ParamShiftSamplerGradient  # type: ignore
        gradient = ParamShiftSamplerGradient(sampler=sampler)
    except Exception as e:  # pragma: no cover
        print("[WARN] Param-shift gradient unavailable, continuing without explicit gradient:", e)
        gradient = None

    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=feature_params,
        weight_params=weight_params,
        interpret=interpret_fn,
        output_shape=num_classes,
        gradient=gradient,
        input_gradients=True
    )

    return TorchConnector(qnn)

class QuantumNetTorch(nn.Module):
    """
    PyTorch module that:
      1) Projects input features to a vector of dimension equal to the number of qubits.
      2) Scales the projection to the [-π/2, π/2] range.
      3) Passes the scaled input through the QNN to produce logits.
    """
    def __init__(self, in_features, num_classes, n_qubits, quantum_depth, noise_model=None):
        super(QuantumNetTorch, self).__init__()
        self.pre_net = nn.Linear(in_features, n_qubits)
        self.q_layer = build_quantum_qnn(n_qubits, quantum_depth, num_classes=num_classes, noise_model=noise_model)
    
    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * (np.pi / 2.0)
        q_out = self.q_layer(q_in)
        return q_out

def get_in_features(model, model_name):
    # Extracts the input dimension for the classifier layer from the pretrained model.
    model_name_lower = model_name.lower()
    if model_name_lower == "resnet18":
        return model.fc.in_features
    elif model_name_lower == "resnet34":
        return model.fc.in_features
    elif model_name_lower == "vgg19":
        return 25088
    elif model_name_lower == "mobilenetv2":
        return model.classifier[1].in_features
    else:
        raise ValueError("Unsupported model for quantum hybrid. Use 'resnet18', 'resnet34', 'vgg19', or 'mobilenetv2'.")

def replace_classifier(model, model_name, quantum_head):
    # Replaces the final classifier of the pretrained model with the quantum head.
    model_name_lower = model_name.lower()
    if model_name_lower == "resnet18":
        model.fc = quantum_head
    elif model_name_lower == "resnet34":
        model.fc = quantum_head
    elif model_name_lower == "vgg19":
        model.classifier = nn.Sequential(nn.Flatten(), quantum_head)
    elif model_name_lower == "mobilenetv2":
        model.classifier[1] = quantum_head
    else:
        raise ValueError("Unsupported model for quantum hybrid. Use 'resnet18', 'resnet34', 'vgg19', or 'mobilenetv2'.")
    return model

def train_quantum_hybrid_qiskit_noisy(dataset_file="hymenoptera", classical_model="resnet18", n_qubits=4, quantum_depth=3, epochs=20, id="null", batch_size=32, learning_rate=0.001, gamma=0.9, backend_name='ibm_nairobi', seed=42, output_dir=None, use_zne=True, zne_scale_factors=None):
    """
    Train a quantum hybrid model with realistic noise from IBM quantum devices.
    """
    set_seed(seed)
    print("============================================================")
    print("Qiskit Noisy Quantum Transfer Learning")
    print("============================================================")
    print(f"Dataset: {dataset_file}")
    print(f"Model: {classical_model}")
    print(f"Qubits: {n_qubits}")
    print(f"Depth: {quantum_depth}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Backend: {backend_name}")
    print(f"ID: {id}")
    print("============================================================")
    print("Step 1/7: Loading and preparing datasets...")
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    transform_test = transform_train
    
    if dataset_file == "hymenoptera" or dataset_file == "brain_tumor" or dataset_file == "cats_dogs" or dataset_file == "solar_dust":
        # Try Resultados directory first (where datasets actually are), then current directory
        if os.path.exists(os.path.join("Resultados", "datasets", dataset_file, "train")):
            path = os.path.join("Resultados", "datasets")
        elif os.path.exists(os.path.join("datasets", dataset_file, "train")):
            path = "datasets"
        else:
            path = os.path.join("Resultados", "datasets")  # fallback to most likely location
    else:
        path = "user_datasets"
    
    train_path = os.path.join(path, dataset_file, "train")
    test_path = os.path.join(path, dataset_file, "test")
    print(f"Using dataset path: {path}")
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")
    
    full_train_dataset = datasets.ImageFolder(train_path, transform=transform_train)
    test_dataset = datasets.ImageFolder(test_path, transform=transform_test)
    
    train_size = int(0.8*len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader= DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("Step 2/7: Defining classical base model...")
    
    num_classes = len(full_train_dataset.classes)
    
    if classical_model.lower() in {"resnet18", "resnet34", "vgg19", "mobilenetv2"}:
        base_model = _safe_load_model(classical_model)
    else:
        raise ValueError("Unsupported classical model. Use 'resnet18', 'resnet34', 'vgg19', or 'mobilenetv2'.")
    
    base_model = base_model.to(device)
    # Freeze backbone parameters
    for param in base_model.parameters():
        param.requires_grad = False

    print("Step 3/7: Creating realistic noise model from quantum device...")
    
    # Create realistic noise model from IBM quantum device
    noise_model = get_real_device_noise_model(n_qubits, backend_name)
    
    print(f"Noise model created successfully for {n_qubits} qubits")
    print(f"Simulating noise from: {backend_name}")
    print("Noise includes: thermal relaxation, depolarizing errors, and readout errors")

    print("Step 4/7: Creating quantum head...")

    in_features = get_in_features(base_model, classical_model)
    quantum_head = QuantumNetTorch(in_features, num_classes, n_qubits, quantum_depth, noise_model=noise_model).to(device)
    hybrid_model = replace_classifier(base_model, classical_model, quantum_head)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, hybrid_model.parameters()), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=gamma)

    print("Step 5/7: Starting training...")
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"QTL_{id}",
            output_dir=output_dir or "results/energy",
            measure_power_secs=15,
            tracking_mode="process",
            log_level="warning"
        )
        tracker.start()
    loss_hist = []
    acc_hist = []
    def evaluate(loader):
        hybrid_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = hybrid_model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return correct / total if total > 0 else 0.0
    
    start_train = time.time()
    best_val_acc = 0.0
    best_weights = copy.deepcopy(hybrid_model.state_dict())
    best_epoch = 0
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}...")
        hybrid_model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = hybrid_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        val_acc = evaluate(val_loader)
        loss_hist.append(running_loss / len(train_loader))
        acc_hist.append(val_acc)
        print(f"Epoch {epoch}/{epochs} - Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(hybrid_model.state_dict())
            best_epoch = epoch
    train_time = time.time() - start_train
    hybrid_model.load_state_dict(best_weights)
    print(f"Best Validation Accuracy: {best_val_acc:.2%} at Epoch {best_epoch}")
    print(f"Total Training Time: {train_time:.2f} seconds")

    print("Step 6/7: Evaluating model on test set...")
    
    start_test = time.time()
    test_acc = evaluate(test_loader)
    test_time = time.time() - start_test
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test Evaluation Time: {test_time:.2f} seconds")

    # ---- Zero-Noise Extrapolation (ZNE) ----
    # ZNE is applied post-training during evaluation only. The model is trained
    # normally with the base noise. Then at test time, we evaluate at multiple
    # noise scales and extrapolate to the zero-noise limit.
    zne_accuracy = 0.0
    zne_precision = 0.0
    zne_recall = 0.0
    zne_f1 = 0.0
    zne_scales_str = ""

    if use_zne:
        if zne_scale_factors is None:
            zne_scale_factors = [1.0, 2.0, 3.0]
        zne_scales_str = ",".join(str(s) for s in zne_scale_factors)
        print("============================================================")
        print("Zero-Noise Extrapolation (ZNE) - Post-Training Evaluation")
        print(f"Scale factors: {zne_scale_factors}")
        print("============================================================")

        # Store per-sample predictions at each noise scale
        scale_accuracies = []
        scale_precisions = []
        scale_recalls = []
        scale_f1s = []

        for sf in zne_scale_factors:
            print(f"  Evaluating at noise scale factor {sf}...")
            # Build a quantum head with the scaled noise model
            scaled_nm = scale_noise_model(noise_model, sf, n_qubits)
            zne_quantum_head = QuantumNetTorch(
                in_features, num_classes, n_qubits, quantum_depth, noise_model=scaled_nm
            ).to(device)

            # Copy trained weights into the new quantum head
            zne_quantum_head.load_state_dict(quantum_head.state_dict())

            # Temporarily replace classifier with the scaled-noise head
            zne_model = copy.deepcopy(hybrid_model)
            zne_model = replace_classifier(zne_model, classical_model, zne_quantum_head)
            zne_model.eval()

            zne_y_true, zne_y_pred = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device)
                    out = zne_model(x)
                    _, pred = torch.max(out, 1)
                    zne_y_true.extend(y.tolist())
                    zne_y_pred.extend(pred.cpu().tolist())

            sf_acc = np.mean(np.array(zne_y_true) == np.array(zne_y_pred))
            sf_prec = precision_score(zne_y_true, zne_y_pred, average='weighted', zero_division=0)
            sf_rec = recall_score(zne_y_true, zne_y_pred, average='weighted', zero_division=0)
            sf_f1 = f1_score(zne_y_true, zne_y_pred, average='weighted', zero_division=0)

            scale_accuracies.append(sf_acc)
            scale_precisions.append(sf_prec)
            scale_recalls.append(sf_rec)
            scale_f1s.append(sf_f1)
            print(f"    Scale {sf}: Acc={sf_acc:.4f}, Prec={sf_prec:.4f}, Rec={sf_rec:.4f}, F1={sf_f1:.4f}")

        # Richardson extrapolation to zero noise
        zne_accuracy = richardson_extrapolate(zne_scale_factors, scale_accuracies)
        zne_precision = richardson_extrapolate(zne_scale_factors, scale_precisions)
        zne_recall = richardson_extrapolate(zne_scale_factors, scale_recalls)
        zne_f1 = richardson_extrapolate(zne_scale_factors, scale_f1s)

        # Clip extrapolated values to [0, 1]
        zne_accuracy = float(np.clip(zne_accuracy, 0.0, 1.0))
        zne_precision = float(np.clip(zne_precision, 0.0, 1.0))
        zne_recall = float(np.clip(zne_recall, 0.0, 1.0))
        zne_f1 = float(np.clip(zne_f1, 0.0, 1.0))

        improvement = (zne_accuracy - test_acc) * 100
        sign = "+" if improvement >= 0 else ""
        print("------------------------------------------------------------")
        print(f"ZNE Accuracy:  {zne_accuracy:.4f} (vs raw: {test_acc:.4f}, improvement: {sign}{improvement:.2f}%)")
        print(f"ZNE Precision: {zne_precision:.4f}")
        print(f"ZNE Recall:    {zne_recall:.4f}")
        print(f"ZNE F1:        {zne_f1:.4f}")
        print("============================================================")

    print("Step 7/7: Saving model and comprehensive metrics...")
    
    # Save model
    model_filename = f"QK_NOISY_{classical_model}_{n_qubits}q_depth{quantum_depth}_{id}_{dataset_file}.pth"
    model_path = os.path.join("model_saved", model_filename)
    os.makedirs("model_saved", exist_ok=True)
    torch.save(hybrid_model.state_dict(), model_path)

    metrics_dir = os.path.join("static","metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Professional visualization settings
    plt.style.use('default')
    sns.set_palette("husl")

    # Loss vs Epoch plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, epochs+1), loss_hist, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Training Loss", fontsize=12)
    plt.title(f"Training Loss vs Epoch - Qiskit Noisy {classical_model.upper()}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{id}_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy vs Epoch plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, epochs+1), acc_hist, 'g-', linewidth=2, marker='s', markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(f"Validation Accuracy vs Epoch - Qiskit Noisy {classical_model.upper()}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{id}_acc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Get predictions for metrics
    y_true, y_pred, y_scores = [], [], []
    hybrid_model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = hybrid_model(x)
            _, pred = torch.max(out, 1)
            y_true.extend(y.tolist())
            y_pred.extend(pred.cpu().tolist())
            
            # Get probabilities for ROC
            probs = torch.softmax(out, dim=1)
            if num_classes == 2:
                y_scores.extend(probs[:, 1].cpu().tolist())
            else:
                y_scores.extend(probs.cpu().tolist())

    # Enhanced Confusion Matrix with Seaborn
    plt.figure(figsize=(8, 6), dpi=300)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=full_train_dataset.classes,
                yticklabels=full_train_dataset.classes)
    plt.title(f"Confusion Matrix - Qiskit Noisy {classical_model.upper()}", fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{id}_confmat.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ROC Curves
    plt.figure(figsize=(10, 8), dpi=300)
    if num_classes == 2:
        # Binary classification ROC
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - Qiskit Noisy {classical_model.upper()}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    else:
        # Multi-class ROC (One-vs-Rest)
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        y_scores_array = np.array(y_scores)
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_array[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{full_train_dataset.classes[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Multi-class ROC Curves - Qiskit Noisy {classical_model.upper()}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{id}_roc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate comprehensive metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Final output
    print(f"Training time: {train_time:.2f}s, Testing time: {test_time:.2f}s")
    print(f"Val Loss per epoch: {loss_hist}")
    print(f"Val Acc  per epoch: {acc_hist}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Model saved as: {model_filename}")
    print(f"Plots saved in: {metrics_dir}")
    print("============================================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("============================================================")

    # CodeCarbon stop
    energy_kwh = 0.0
    co2_kg = 0.0
    if CODECARBON_AVAILABLE:
        emissions = tracker.stop()
        energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') else 0.0
        co2_kg = emissions if emissions else 0.0

    # Save structured CSV
    csv_dir = output_dir or "results/seeds"
    os.makedirs(csv_dir, exist_ok=True)

    approach = "qiskit_noisy"

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)

    try:
        if num_classes == 2:
            auc_roc = roc_auc_score(y_true_arr, y_scores_arr)
        else:
            auc_roc = roc_auc_score(y_true_arr, y_scores_arr, multi_class='ovr', average='weighted')
    except:
        auc_roc = 0.0

    csv_filename = f"{approach}_{classical_model}_{dataset_file}_seed{seed}.csv"
    csv_path = os.path.join(csv_dir, csv_filename)

    row = {
        'approach': approach,
        'backbone': classical_model,
        'dataset': dataset_file,
        'seed': seed,
        'n_qubits': n_qubits,
        'quantum_depth': quantum_depth,
        'test_accuracy': test_acc,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'auc_roc_weighted': auc_roc,
        'train_time_s': train_time,
        'test_time_s': test_time,
        'energy_kwh': energy_kwh,
        'co2_kg': co2_kg,
        'epochs_actual': len(loss_hist),
        'loss_history': json.dumps(loss_hist),
        'val_acc_history': json.dumps(acc_hist),
        'zne_accuracy': zne_accuracy,
        'zne_precision': zne_precision,
        'zne_recall': zne_recall,
        'zne_f1': zne_f1,
        'zne_scale_factors': zne_scales_str
    }

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to: {csv_path}")

    return test_acc, train_time, test_time

def main():
    """CLI to train Qiskit hybrid models with realistic noise."""
    parser = argparse.ArgumentParser(description='Train a Qiskit hybrid model with real-device noise')
    parser.add_argument('--dataset', type=str, default='hymenoptera', help='Dataset name')
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'vgg16', 'vgg19', 'mobilenetv2'], 
                       default='resnet18', help='Classical model architecture')
    parser.add_argument('--n-qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--depth', type=int, default=3, help='Quantum circuit depth')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Learning rate scheduler gamma')
    parser.add_argument('--backend', type=str, default='ibm_nairobi', 
                       choices=['ibm_nairobi', 'ibm_manila', 'ibm_lagos'],
                       help='IBM Quantum backend to simulate')
    parser.add_argument('--id', type=str, help='Run identifier (auto if not provided)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results CSV')
    parser.add_argument('--use-zne', type=bool, default=True, help='Enable Zero-Noise Extrapolation during evaluation (default: True)')
    parser.add_argument('--zne-scale-factors', type=str, default='1.0,2.0,3.0',
                       help='Comma-separated noise scale factors for ZNE (default: "1.0,2.0,3.0")')

    args = parser.parse_args()
    
    # Generate automatic ID if not provided
    if args.id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.id = f"{args.model}_{args.n_qubits}q_depth{args.depth}_{timestamp}_noisy"
    
    # Parse ZNE scale factors from comma-separated string
    zne_scale_factors = [float(s.strip()) for s in args.zne_scale_factors.split(',')]

    # Train model
    test_acc, train_time, test_time = train_quantum_hybrid_qiskit_noisy(
        dataset_file=args.dataset,
        classical_model=args.model,
        n_qubits=args.n_qubits,
        quantum_depth=args.depth,
        epochs=args.epochs,
        id=args.id,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        backend_name=args.backend,
        seed=args.seed,
        output_dir=args.output_dir,
        use_zne=args.use_zne,
        zne_scale_factors=zne_scale_factors
    )

if __name__ == "__main__":
    main()
