"""Qiskit quantum head (ideal and noisy)."""

import torch
import torch.nn as nn
import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler

try:
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.connectors import TorchConnector
except ImportError as e:
    raise ImportError(f"qiskit-machine-learning required: {e}")


def _build_circuit(n_qubits, depth):
    """Build parameterized quantum circuit."""
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.h(i)

    feature_params = [Parameter(f"theta_{i}") for i in range(n_qubits)]
    for i in range(n_qubits):
        qc.ry(feature_params[i], i)

    weight_params = []
    for layer in range(depth):
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        lp = [Parameter(f"phi_{layer}_{i}") for i in range(n_qubits)]
        weight_params.extend(lp)
        for i in range(n_qubits):
            qc.ry(lp[i], i)

    return qc, feature_params, weight_params


def _build_qnn(n_qubits, depth, num_classes, shots=None, noise=False, noise_params=None):
    """Build SamplerQNN with optional noise."""
    qc, feature_params, weight_params = _build_circuit(n_qubits, depth)

    def interpret_index(bit_int):
        return bin(bit_int).count("1") % num_classes

    sampler = None
    if noise or shots is not None:
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import Sampler as AerSampler
            from qiskit_aer.noise import NoiseModel, depolarizing_error

            backend_opts = {}
            if noise and noise_params:
                nm = NoiseModel()
                p1q = noise_params.get("p1q", 0.0002)
                p2q = noise_params.get("p2q", 0.005)
                if p1q > 0:
                    e1 = depolarizing_error(p1q, 1)
                    for g in ["x", "y", "z", "rx", "ry", "rz", "h", "sx", "id"]:
                        try:
                            nm.add_all_qubit_quantum_error(e1, [g])
                        except Exception:
                            pass
                if p2q > 0:
                    e2 = depolarizing_error(p2q, 2)
                    for g in ["cx", "cz", "swap"]:
                        try:
                            nm.add_all_qubit_quantum_error(e2, [g])
                        except Exception:
                            pass
                backend_opts["noise_model"] = nm

            backend = AerSimulator(**backend_opts)
            sampler = AerSampler()
            sampler.set_options(backend=backend)
            sampler.set_options(shots=shots or 1024)
        except Exception:
            sampler = Sampler()
    else:
        sampler = Sampler()

    gradient = None
    try:
        from qiskit_machine_learning.gradients import ParamShiftSamplerGradient
        gradient = ParamShiftSamplerGradient(sampler=sampler)
    except Exception:
        pass

    qnn = SamplerQNN(
        circuit=qc,
        sampler=sampler,
        input_params=feature_params,
        weight_params=weight_params,
        interpret=interpret_index,
        output_shape=num_classes,
        gradient=gradient,
        input_gradients=True,
    )
    return TorchConnector(qnn)


class QiskitHead(nn.Module):
    """Hybrid head: Linear -> tanh*pi/2 -> Qiskit QNN."""

    def __init__(self, feature_dim, num_classes, n_qubits=4, depth=3,
                 noise=False, noise_params=None, shots=None):
        super().__init__()
        self.pre = nn.Linear(feature_dim, n_qubits)
        self.qlayer = _build_qnn(n_qubits, depth, num_classes,
                                  shots=shots, noise=noise, noise_params=noise_params)

    def forward(self, x):
        z = self.pre(x)
        q_input = torch.tanh(z) * (np.pi / 2.0)
        return self.qlayer(q_input)
