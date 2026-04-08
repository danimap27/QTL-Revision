"""PennyLane quantum head (ideal and noisy)."""

import torch
import torch.nn as nn
import numpy as np
import pennylane as qml


def _create_ideal_circuit(n_qubits, depth, n_outputs):
    """Create ideal PennyLane quantum layer."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(w)) for w in range(n_outputs)]

    weight_shapes = {"weights": (depth, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


def _create_noisy_circuit(n_qubits, depth, n_outputs, noise_params):
    """Create noisy PennyLane quantum layer with IBM-calibrated noise."""
    dev = qml.device("default.mixed", wires=n_qubits)

    # Extract noise parameters (Heron r2 defaults)
    T1 = noise_params.get("T1_us", 250) * 1e-6   # convert us -> s
    T2 = noise_params.get("T2_us", 150) * 1e-6
    t1q = noise_params.get("t1q_ns", 32) * 1e-9   # single-qubit gate time
    t2q = noise_params.get("t2q_ns", 68) * 1e-9   # two-qubit gate time
    p1q = noise_params.get("p1q", 0.0002)
    p2q = noise_params.get("p2q", 0.005)
    readout_err = noise_params.get("readout_error", 0.012)

    # Derived damping probabilities
    gamma_1q = 1 - np.exp(-t1q / T1) if T1 > 0 else 0
    gamma_2q = 1 - np.exp(-t2q / T1) if T1 > 0 else 0
    phase_1q = 1 - np.exp(-t1q / T2) if T2 > 0 else 0
    phase_2q = 1 - np.exp(-t2q / T2) if T2 > 0 else 0

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # Encode inputs
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
            qml.AmplitudeDamping(gamma_1q, wires=i)
            qml.PhaseDamping(phase_1q, wires=i)

        # Variational layers with noise
        for layer in range(depth):
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                for q in (i, i + 1):
                    qml.AmplitudeDamping(gamma_2q, wires=q)
                    qml.PhaseDamping(phase_2q, wires=q)
            for i in range(n_qubits):
                qml.RY(weights[layer][i], wires=i)
                qml.AmplitudeDamping(gamma_1q, wires=i)
                qml.PhaseDamping(phase_1q, wires=i)

        # Readout noise: bit-flip before measurement approximation
        for i in range(n_qubits):
            qml.BitFlip(readout_err, wires=i)

        return [qml.expval(qml.PauliZ(w)) for w in range(n_outputs)]

    weight_shapes = {"weights": (depth, n_qubits)}
    return qml.qnn.TorchLayer(circuit, weight_shapes)


class PennyLaneHead(nn.Module):
    """Hybrid head: Linear -> tanh*pi/2 -> VQC -> Linear.

    Works for both ideal and noisy PennyLane backends.
    """

    def __init__(self, feature_dim, num_classes, n_qubits=4, depth=3,
                 noise=False, noise_params=None, backend="default.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.pre = nn.Linear(feature_dim, n_qubits)
        if noise and noise_params:
            self.qlayer = _create_noisy_circuit(n_qubits, depth, num_classes, noise_params)
        else:
            self.qlayer = _create_ideal_circuit(n_qubits, depth, num_classes)

    def forward(self, x):
        x = self.pre(x)
        x = torch.tanh(x) * (np.pi / 2)
        if x.dim() == 1:
            return self.qlayer(x)
        # Process sample-by-sample (QNode batch limitation)
        return torch.stack([self.qlayer(xi) for xi in x], dim=0)
