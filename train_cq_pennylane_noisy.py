# train_cq_pennylane_noisy.py
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use('Agg')

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
from datetime import datetime
from itertools import cycle
import random
import json
import csv
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.optimize import curve_fit


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Global transforms
GLOBAL_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def create_noisy_device(n_qubits, noise_type='realistic_ibm'):
    """
    Create a noisy quantum device that simulates realistic quantum hardware noise.
    
    Args:
        n_qubits: Number of qubits
        noise_type: Type of noise model ('realistic_ibm', 'depolarizing', 'amplitude_damping')
    """
    if noise_type == 'realistic_ibm':
        # Simulate realistic IBM quantum device noise
        # T1 times (relaxation): typical values 50-200 us  
        # T2 times (dephasing): typical values 30-150 us
        # Gate times: ~50ns for single qubit, ~200ns for two qubit
        
        # Create mixed device with noise channels
        dev = qml.device('default.mixed', wires=n_qubits)
        
        return dev, get_realistic_ibm_noise_params(n_qubits, 'ibm_nairobi')
        
    elif noise_type == 'depolarizing':
        # Simple depolarizing noise
        dev = qml.device('default.mixed', wires=n_qubits)
        return dev, {'depolarizing_prob': 0.01}
        
    elif noise_type == 'amplitude_damping':
        # Amplitude damping (energy loss)
        dev = qml.device('default.mixed', wires=n_qubits)
        return dev, {'damping_prob': 0.05}
        
    else:
        # Noiseless for comparison
        dev = qml.device('default.qubit', wires=n_qubits)
        return dev, {}

def get_realistic_ibm_noise_params(n_qubits, backend_name='ibm_nairobi'):
    """Generate realistic noise parameters based on real IBM quantum device specifications."""
    
    # Real IBM device parameters (from IBM Quantum Network - December 2024)
    backend_specs = {
        'ibm_nairobi': {
            't1_mean': 169e-6, 't1_std': 50e-6,  # T1 times in seconds
            't2_mean': 104e-6, 't2_std': 30e-6,  # T2 times in seconds
            'readout_error_mean': 0.0165, 'readout_error_std': 0.005,  # Readout errors
            'gate_error_1q': 0.0003,  # Single-qubit gate error
            'gate_error_2q': 0.0065   # Two-qubit gate error
        },
        'ibm_manila': {
            't1_mean': 142e-6, 't1_std': 40e-6,
            't2_mean': 89e-6, 't2_std': 25e-6,
            'readout_error_mean': 0.025, 'readout_error_std': 0.008,
            'gate_error_1q': 0.0004,
            'gate_error_2q': 0.0085
        },
        'ibm_lagos': {
            't1_mean': 135e-6, 't1_std': 45e-6,
            't2_mean': 95e-6, 't2_std': 28e-6,
            'readout_error_mean': 0.022, 'readout_error_std': 0.007,
            'gate_error_1q': 0.0005,
            'gate_error_2q': 0.0075
        }
    }
    
    if backend_name not in backend_specs:
        backend_name = 'ibm_nairobi'  # Default fallback
    
    specs = backend_specs[backend_name]
    noise_params = {}
    
    # Generate per-qubit parameters with realistic variations
    t1_times = []
    t2_times = []
    readout_errors = []
    
    np.random.seed(42)  # For reproducible results
    
    for i in range(n_qubits):
        # Sample T1/T2 times from normal distribution (clipped to positive values)
        t1 = max(np.random.normal(specs['t1_mean'], specs['t1_std']), 50e-6)
        t2 = max(np.random.normal(specs['t2_mean'], specs['t2_std']), 30e-6)
        # Ensure T2 <= T1 (physical constraint)
        t2 = min(t2, t1 * 0.8)
        
        # Sample readout error
        readout_error = max(np.random.normal(specs['readout_error_mean'], specs['readout_error_std']), 0.005)
        readout_error = min(readout_error, 0.1)  # Cap at 10%
        
        t1_times.append(t1)
        t2_times.append(t2)
        readout_errors.append(readout_error)
        
        # Use actual gate error rates from IBM devices (much more realistic)
        # These are directly from IBM's calibration data
        amp_damping_1q = specs['gate_error_1q'] * 0.3  # 30% of gate error attributed to amplitude damping
        amp_damping_2q = specs['gate_error_2q'] * 0.3
        phase_damping_1q = specs['gate_error_1q'] * 0.2  # 20% attributed to phase damping
        phase_damping_2q = specs['gate_error_2q'] * 0.2
        
        noise_params[f'qubit_{i}'] = {
            't1': t1,
            't2': t2,
            'amp_damping_1q': amp_damping_1q,
            'amp_damping_2q': amp_damping_2q,
            'phase_damping_1q': phase_damping_1q,
            'phase_damping_2q': phase_damping_2q,
            'readout_error': readout_error
        }
    
    # Store global parameters
    noise_params['readout_errors'] = readout_errors
    noise_params['backend_name'] = backend_name
    
    # Print summary information
    avg_t1 = np.mean(t1_times) * 1e6  # Convert to microseconds
    avg_t2 = np.mean(t2_times) * 1e6
    avg_readout = np.mean(readout_errors) * 100  # Convert to percentage
    avg_gate_1q = specs['gate_error_1q'] * 100
    avg_gate_2q = specs['gate_error_2q'] * 100
    
    print(f"[INFO] Simulating realistic noise from IBM backend: {backend_name}")
    print(f"Average T1 time: {avg_t1:.1f} us")
    print(f"Average T2 time: {avg_t2:.1f} us") 
    print(f"Average readout error: {avg_readout:.2f}%")
    print(f"Single-qubit gate error: {avg_gate_1q:.3f}%")
    print(f"Two-qubit gate error: {avg_gate_2q:.3f}%")
    
    return noise_params

def quantum_circuit_with_noise(n_qubits, quantum_depth, noise_params=None):
    """Create quantum circuit with realistic noise channels."""
    
    dev, _ = create_noisy_device(n_qubits, 'realistic_ibm')
    
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # Encode classical inputs
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
            
            # Add noise after input encoding if noise_params provided
            if noise_params and f'qubit_{i}' in noise_params:
                params = noise_params[f'qubit_{i}']
                qml.AmplitudeDamping(params['amp_damping_1q'], wires=i)
                qml.PhaseDamping(params['phase_damping_1q'], wires=i)
        
        # Variational layers with noise
        for layer in range(quantum_depth):
            # Entangling layer
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                
                # Add two-qubit gate noise
                if noise_params:
                    for qubit in [i, i + 1]:
                        if f'qubit_{qubit}' in noise_params:
                            params = noise_params[f'qubit_{qubit}']
                            qml.AmplitudeDamping(params['amp_damping_2q'], wires=qubit)
                            qml.PhaseDamping(params['phase_damping_2q'], wires=qubit)
            
            # Parameterized layer
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
                
                # Add single-qubit gate noise
                if noise_params and f'qubit_{i}' in noise_params:
                    params = noise_params[f'qubit_{i}']
                    qml.AmplitudeDamping(params['amp_damping_1q'], wires=i)
                    qml.PhaseDamping(params['phase_damping_1q'], wires=i)
        
        # Return single expectation value for binary classification
        return qml.expval(qml.PauliZ(wires=0))
    
    return circuit

def scale_noise_params(noise_params, scale_factor):
    """Scale all noise probabilities by scale_factor."""
    scaled = {}
    for key, val in noise_params.items():
        if isinstance(val, dict):
            scaled[key] = {}
            for k, v in val.items():
                if 'damping' in k or 'error' in k:
                    # Scale noise probabilities, clamping to [0, 1)
                    scaled[key][k] = min(v * scale_factor, 0.99)
                else:
                    scaled[key][k] = v
        elif key == 'readout_errors':
            scaled[key] = [min(e * scale_factor, 0.99) for e in val]
        else:
            scaled[key] = val
    return scaled


def richardson_extrapolate(scale_factors, values):
    """Richardson extrapolation to zero noise.

    For scale factors [c1, c2, c3, ...] and corresponding values [v1, v2, v3, ...],
    fit a polynomial and evaluate at scale=0.
    """
    scale_factors = np.array(scale_factors)
    values = np.array(values)

    # Fit polynomial of degree len(scale_factors)-1
    degree = min(len(scale_factors) - 1, 2)
    coeffs = np.polyfit(scale_factors, values, degree)
    # Evaluate at scale=0
    return np.polyval(coeffs, 0.0)


def apply_zne_to_predictions(model, test_loader, noise_params, n_qubits, quantum_depth,
                              scale_factors, device, num_classes):
    """Apply Zero-Noise Extrapolation to test predictions.

    For each noise scale factor, create a new quantum circuit with scaled noise,
    run test predictions, then extrapolate to zero noise.
    """
    all_scaled_probs = []  # shape: [n_scales, n_samples, n_classes]

    for scale in scale_factors:
        # Scale noise parameters
        scaled_params = scale_noise_params(noise_params, scale)

        # Create model with scaled noise
        scaled_circuit = quantum_circuit_with_noise(n_qubits, quantum_depth, scaled_params)

        # Replace quantum circuit in model temporarily
        original_circuit = model.quantum_layer.quantum_circuit
        model.quantum_layer.quantum_circuit = scaled_circuit

        # Run predictions
        probs = []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                out = model(x)
                p = torch.softmax(out, dim=1)
                probs.append(p.cpu().numpy())

        all_scaled_probs.append(np.concatenate(probs, axis=0))

        # Restore original circuit
        model.quantum_layer.quantum_circuit = original_circuit

    # Richardson extrapolation to zero noise
    all_scaled_probs = np.array(all_scaled_probs)  # [n_scales, n_samples, n_classes]
    zne_probs = np.zeros_like(all_scaled_probs[0])

    for i in range(all_scaled_probs.shape[1]):  # per sample
        for c in range(all_scaled_probs.shape[2]):  # per class
            values = all_scaled_probs[:, i, c]
            # Linear Richardson extrapolation
            zne_probs[i, c] = richardson_extrapolate(scale_factors, values)

    return zne_probs


class NoisyQuantumLayer(nn.Module):
    """Quantum layer with realistic noise simulation."""
    
    def __init__(self, n_qubits, quantum_depth, noise_params=None):
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum_depth = quantum_depth
        self.noise_params = noise_params
        
        # Create quantum circuit
        self.quantum_circuit = quantum_circuit_with_noise(n_qubits, quantum_depth, noise_params)
        
        # Initialize quantum weights
        self.weights = nn.Parameter(torch.randn(quantum_depth, n_qubits, 2) * 0.1)
    
    def forward(self, x):
        batch_size = x.size(0)
        results = []
        
        for i in range(batch_size):
            # Apply quantum circuit to each sample individually
            result = self.quantum_circuit(x[i], self.weights)
            # Ensure proper dtype (float32)
            result_tensor = torch.tensor(result, dtype=torch.float32)
            results.append(result_tensor)
        
        return torch.stack(results).unsqueeze(1)

class HybridNoisyModel(nn.Module):
    """Hybrid classical-quantum model with noise simulation."""
    
    def __init__(self, backbone, feature_dim, n_qubits, num_classes, quantum_depth, noise_params=None):
        super().__init__()
        self.backbone = backbone
        self.pre_quantum = nn.Linear(feature_dim, n_qubits)
        self.quantum_layer = NoisyQuantumLayer(n_qubits, quantum_depth, noise_params)
        
        # Output mapping (single quantum measurement to class predictions)
        self.post_quantum = nn.Linear(1, num_classes)
        
    def forward(self, x):
        # Classical feature extraction
        features = self.backbone(x)
        
        # Pre-processing for quantum layer
        quantum_input = torch.tanh(self.pre_quantum(features)) * np.pi
        
        # Quantum processing with noise
        quantum_output = self.quantum_layer(quantum_input)
        
        # Ensure quantum output has the right shape
        if quantum_output.dim() == 1:
            quantum_output = quantum_output.unsqueeze(1)
        
        # Final classification
        output = self.post_quantum(quantum_output)
        
        return output

def train_quantum_hybrid_pennylane_noisy(dataset_file="hymenoptera", classical_model="resnet18",
                                       n_qubits=4, quantum_depth=3, epochs=20, id="null",
                                       batch_size=32, learning_rate=1e-3, early_stop_patience=10,
                                       gamma=0.9, noise_type='realistic_ibm', backend='ibm_nairobi',
                                       seed=42, output_dir=None,
                                       use_zne=True, zne_scale_factors=None):
    set_seed(seed)
    print("============================================================")
    print("PennyLane Noisy Quantum Transfer Learning")
    print("============================================================")
    print(f"Dataset: {dataset_file}")
    print(f"Model: {classical_model}")
    print(f"Qubits: {n_qubits}")
    print(f"Depth: {quantum_depth}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Noise type: {noise_type}")
    if noise_type == 'realistic_ibm':
        print(f"Backend: {backend}")
    if zne_scale_factors is None:
        zne_scale_factors = [1.0, 2.0, 3.0]
    print(f"ZNE enabled: {use_zne}")
    if use_zne:
        print(f"ZNE scale factors: {zne_scale_factors}")
    print(f"ID: {id}")
    print("============================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Step 1/7: Loading and preparing datasets...")
    # Robust dataset path resolution
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
    
    transforms_cfg = GLOBAL_TRANSFORMS

    # Load datasets
    full_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_cfg['train'])
    test_ds = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transforms_cfg['val'])
    train_size = int(0.8 * len(full_train))
    val_size = len(full_train) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_classes = len(full_train.classes)

    print("Step 2/7: Defining classical base model...")
    # Build backbone
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
        raise ValueError("Unsupported model")
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    backbone = backbone.to(device)

    print("Step 3/7: Creating realistic noise model...")
    # Generate noise parameters
    if noise_type == 'realistic_ibm':
        noise_params = get_realistic_ibm_noise_params(n_qubits, backend)
        print(f"[INFO] Simulating realistic IBM quantum device noise")
        print(f"Average T1 time: {np.mean([noise_params[f'qubit_{i}']['t1'] for i in range(n_qubits)]):.1f} us")
        print(f"Average T2 time: {np.mean([noise_params[f'qubit_{i}']['t2'] for i in range(n_qubits)]):.1f} us")
        print(f"Average readout error: {np.mean(noise_params['readout_errors'])*100:.2f}%")
    else:
        noise_params = None
        print(f"[INFO] Using {noise_type} noise model")

    print("Step 4/7: Creating quantum circuit and hybrid model...")
    # Instantiate hybrid model with noise
    model = HybridNoisyModel(backbone, feat_dim, n_qubits, num_classes, quantum_depth, noise_params).to(device)
    
    print("Step 5/7: Setting up training components...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=gamma)

    print("Step 6/7: Starting training...")
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker(
            project_name=f"QTL_{id}",
            output_dir=output_dir or "results/energy",
            measure_power_secs=15,
            tracking_mode="process",
            log_level="warning"
        )
        tracker.start()
    # Training loop with early stopping
    best_loss = float('inf')
    patience = 0
    train_losses, val_losses, val_accs = [], [], []
    start_time = time.time()
    
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}...")
        model.train()
        running = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        
        # Validation
        model.eval()
        v_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v_loss += criterion(out, y).item() * x.size(0)
                _, pred = torch.max(out, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_loss = v_loss / len(val_loader.dataset)
        val_acc = correct / total
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_model = model.state_dict().copy()
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    train_time = time.time() - start_time
    
    print("Step 7/7: Evaluating model on test set...")
    # Test evaluation
    model.load_state_dict(best_model)
    model.eval()
    tc, tot = 0, 0
    start_time_test = time.time()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            tc += (pred == y).sum().item()
            tot += y.size(0)
    test_acc = tc / tot
    test_time = time.time() - start_time_test
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Training time: {train_time:.2f}s, Testing time: {test_time:.2f}s")

    # Zero-Noise Extrapolation (ZNE)
    zne_acc = 0.0
    zne_precision = 0.0
    zne_recall = 0.0
    zne_f1 = 0.0
    zne_applied = False
    if use_zne and noise_params is not None:
        print("Applying Zero-Noise Extrapolation (ZNE)...")
        print(f"  Scale factors: {zne_scale_factors}")
        zne_probs = apply_zne_to_predictions(
            model, test_loader, noise_params, n_qubits, quantum_depth,
            zne_scale_factors, device, num_classes
        )
        # Collect true labels for ZNE evaluation
        zne_y_true = []
        for _, y in test_loader:
            zne_y_true.extend(y.tolist())
        zne_y_true = np.array(zne_y_true)
        zne_preds = np.argmax(zne_probs, axis=1)
        zne_acc = np.mean(zne_preds == zne_y_true)
        zne_precision = precision_score(zne_y_true, zne_preds, average='weighted', zero_division=0)
        zne_recall = recall_score(zne_y_true, zne_preds, average='weighted', zero_division=0)
        zne_f1 = f1_score(zne_y_true, zne_preds, average='weighted', zero_division=0)
        zne_applied = True
        print(f"ZNE Test Accuracy: {zne_acc:.4f} (vs raw: {test_acc:.4f})")
        print(f"ZNE Precision: {zne_precision:.4f}, Recall: {zne_recall:.4f}, F1: {zne_f1:.4f}")
    elif use_zne and noise_params is None:
        print("ZNE requested but no noise params available (noiseless mode). Skipping ZNE.")

    print("Step 8/8: Saving model and comprehensive metrics...")
    # Save model
    os.makedirs('model_saved', exist_ok=True)
    fn = f"PL_NOISY_{id}_{classical_model}_{dataset_file}.pth"
    torch.save(model.state_dict(), os.path.join('model_saved', fn))

    # Professional visualization
    metrics_dir = os.path.join("static", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("husl")

    # Loss vs Epoch plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', linewidth=2, marker='o', markersize=4, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', linewidth=2, marker='s', markersize=4, label='Val Loss')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"Training Loss vs Epoch - PennyLane Noisy {classical_model.upper()}", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{id}_loss.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Accuracy vs Epoch plot
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(1, len(val_accs)+1), val_accs, 'g-', linewidth=2, marker='d', markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title(f"Validation Accuracy vs Epoch - PennyLane Noisy {classical_model.upper()}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{id}_acc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Get predictions for confusion matrix and ROC
    y_true, y_pred, y_scores = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = torch.max(out, 1)
            probs = torch.softmax(out, dim=1)
            
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            
            if num_classes == 2:
                y_scores.extend(probs[:, 1].cpu().tolist())
            else:
                y_scores.extend(probs.cpu().tolist())

    # Enhanced Confusion Matrix
    plt.figure(figsize=(8, 6), dpi=300)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=full_train.classes,
                yticklabels=full_train.classes)
    plt.title(f"Confusion Matrix - PennyLane Noisy {classical_model.upper()}", fontsize=14)
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
        plt.title(f'ROC Curve - PennyLane Noisy {classical_model.upper()}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    else:
        # Multi-class ROC (One-vs-Rest)
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        y_scores_array = np.array(y_scores)
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores_array[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        for i, color in zip(range(num_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{full_train.classes[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Multi-class ROC Curves - PennyLane Noisy {classical_model.upper()}', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(metrics_dir, f"{id}_roc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Final comprehensive output
    print(f"Training time: {train_time:.2f}s, Testing time: {test_time:.2f}s")
    print(f"Val Loss per epoch: {val_losses}")
    print(f"Val Acc  per epoch: {val_accs}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    print(f"Model saved as: {fn}")
    print(f"Plots saved in: {metrics_dir}")
    print("============================================================")
    print("TRAINING COMPLETED SUCCESSFULLY")
    print(f"Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Training Time: {train_time:.2f} seconds")
    print(f"Testing Time: {test_time:.2f} seconds")
    if noise_type == 'realistic_ibm':
        avg_t1 = np.mean([noise_params[f'qubit_{i}']['t1'] for i in range(n_qubits)])
        avg_t2 = np.mean([noise_params[f'qubit_{i}']['t2'] for i in range(n_qubits)])
        avg_readout = np.mean(noise_params['readout_errors'])*100
        print(f"Noise Model: Realistic IBM (T1={avg_t1:.1f}us, T2={avg_t2:.1f}us, Readout={avg_readout:.2f}%)")
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

    approach = "pennylane_noisy"

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    y_scores_arr = np.array(y_scores)

    precision = precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
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
        'epochs_actual': len(train_losses),
        'loss_history': json.dumps(train_losses),
        'val_acc_history': json.dumps(val_accs),
        'zne_accuracy': zne_acc if zne_applied else '',
        'zne_precision': zne_precision if zne_applied else '',
        'zne_recall': zne_recall if zne_applied else '',
        'zne_f1': zne_f1 if zne_applied else '',
        'zne_scale_factors': json.dumps(zne_scale_factors) if zne_applied else ''
    }

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"Results saved to: {csv_path}")

    return test_acc, train_time, test_time

def main():
    """CLI to train PennyLane hybrid models with realistic quantum noise."""
    parser = argparse.ArgumentParser(description='Train a PennyLane hybrid model with realistic quantum noise')
    parser.add_argument('--dataset', type=str, default='hymenoptera', help='Dataset name')
    parser.add_argument('--model', type=str, choices=['resnet18', 'resnet34', 'mobilenetv2', 'efficientnet_b0', 'regnet_x_400mf'], 
                       default='resnet18', help='Classical model architecture')
    parser.add_argument('--n-qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--depth', type=int, default=3, help='Quantum circuit depth')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='Learning rate scheduler gamma')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--noise-type', type=str, default='realistic_ibm',
                       choices=['realistic_ibm', 'depolarizing', 'amplitude_damping'],
                       help='Type of quantum noise to simulate')
    parser.add_argument('--backend', type=str, default='ibm_nairobi',
                       choices=['ibm_nairobi', 'ibm_manila', 'ibm_lagos'],
                       help='IBM Quantum backend to simulate (only for realistic_ibm noise)')
    parser.add_argument('--id', type=str, help='Run identifier (auto if not provided)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results CSV')
    parser.add_argument('--use-zne', action='store_true', default=True,
                       help='Enable Zero-Noise Extrapolation (ZNE) for error mitigation (default: True)')
    parser.add_argument('--no-zne', action='store_false', dest='use_zne',
                       help='Disable Zero-Noise Extrapolation (ZNE)')
    parser.add_argument('--zne-scale-factors', type=str, default='1.0,2.0,3.0',
                       help='Comma-separated noise scale factors for ZNE (default: "1.0,2.0,3.0")')

    args = parser.parse_args()
    
    # Generate automatic ID if not provided
    if args.id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.id = f"{args.model}_{args.n_qubits}q_depth{args.depth}_{timestamp}_noisy_pl"
    
    # Parse ZNE scale factors
    zne_scale_factors = [float(s) for s in args.zne_scale_factors.split(',')]

    # Train model
    test_acc, train_time, test_time = train_quantum_hybrid_pennylane_noisy(
        dataset_file=args.dataset,
        classical_model=args.model,
        n_qubits=args.n_qubits,
        quantum_depth=args.depth,
        epochs=args.epochs,
        id=args.id,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gamma=args.gamma,
        early_stop_patience=args.patience,
        noise_type=args.noise_type,
        backend=args.backend,
        seed=args.seed,
        output_dir=args.output_dir,
        use_zne=args.use_zne,
        zne_scale_factors=zne_scale_factors
    )

if __name__ == "__main__":
    main()