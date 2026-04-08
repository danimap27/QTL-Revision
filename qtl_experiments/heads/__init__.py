"""Head factory: instantiate any head from config."""

from .linear_head import LinearHead
from .mlp_a_head import MLPAHead
from .mlp_b_head import MLPBHead
from .pennylane_head import PennyLaneHead
from .qiskit_head import QiskitHead


def create_head(head_cfg, feature_dim, num_classes):
    """Create a head module from its config dict.

    Args:
        head_cfg: dict from config.yaml heads list entry
        feature_dim: int, backbone output dimension
        num_classes: int

    Returns:
        nn.Module (the head), head_type str
    """
    name = head_cfg["name"]
    head_type = head_cfg["type"]

    if name == "linear":
        return LinearHead(feature_dim, num_classes), head_type

    elif name == "mlp_a":
        hidden_dim = head_cfg.get("hidden_dim", 4)
        return MLPAHead(feature_dim, num_classes, hidden_dim), head_type

    elif name == "mlp_b":
        hidden_dims = head_cfg.get("hidden_dims", [128, 64])
        return MLPBHead(feature_dim, num_classes, hidden_dims), head_type

    elif head_type == "pennylane":
        n_qubits = head_cfg.get("n_qubits", 4)
        depth = head_cfg.get("depth", 3)
        noise = head_cfg.get("noise", False)
        noise_params = head_cfg.get("noise_params", None)
        backend = head_cfg.get("backend", "default.qubit")
        return PennyLaneHead(
            feature_dim, num_classes, n_qubits, depth,
            noise=noise, noise_params=noise_params, backend=backend
        ), head_type

    elif head_type == "qiskit":
        n_qubits = head_cfg.get("n_qubits", 4)
        depth = head_cfg.get("depth", 3)
        noise = head_cfg.get("noise", False)
        noise_params = head_cfg.get("noise_params", None)
        shots = head_cfg.get("shots", None)
        return QiskitHead(
            feature_dim, num_classes, n_qubits, depth,
            noise=noise, noise_params=noise_params, shots=shots
        ), head_type

    else:
        raise ValueError(f"Unknown head: {name} (type={head_type})")
