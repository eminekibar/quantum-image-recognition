"""
Hibrit Kuantum-Klasik Sinir Ağı (QNN)
--------------------------------------
Mimari:
  Klasik CNN Encoder (28x28 -> 4 ozellik)
  -> Kuantum Devre (PennyLane, 4 qubit)
  -> Klasik Siniflandirici (4 -> 10 sinif)
"""

import pennylane as qml
import torch
import torch.nn as nn
import yaml
import os


def load_config(path: str = None) -> dict:
    if path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_quantum_device(n_qubits: int):
    return qml.device("default.qubit", wires=n_qubits)


def make_quantum_circuit(n_qubits: int, n_layers: int):
    dev = build_quantum_device(n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        # Veriyi qubit'lere kodla
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        # Ogrenilir kuantum katmanlari
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        # Her qubit icin beklenti degeri olc
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


class QuantumLayer(nn.Module):
    """PennyLane devresini PyTorch katmani olarak sarar."""

    def __init__(self, n_qubits: int, n_layers: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit = make_quantum_circuit(n_qubits, n_layers)

        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.weights = nn.Parameter(torch.randn(weight_shape) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        outputs = []
        for sample in x:
            result = self.circuit(sample, self.weights)
            outputs.append(torch.stack(result))
        return torch.stack(outputs).float()  # (batch, n_qubits) — float32'ye donustur


class ClassicalEncoder(nn.Module):
    """28x28 goruntu -> n_qubits boyutlu ozellik vektoru."""

    def __init__(self, n_qubits: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 14x14
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                              # 7x7
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_qubits),
            nn.Tanh(),   # Kuantum girisini [-1, 1] araligina sikistir
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


class HybridQNN(nn.Module):
    """
    Tam Hibrit Kuantum-Klasik Model.

    Kullanim:
        model = HybridQNN.from_config()
        output = model(images)  # images: (batch, 1, 28, 28)
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3, n_classes: int = 10):
        super().__init__()
        self.encoder = ClassicalEncoder(n_qubits)
        self.quantum = QuantumLayer(n_qubits, n_layers)
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)       # (batch, n_qubits)
        x = self.quantum(x)        # (batch, n_qubits)
        x = self.classifier(x)     # (batch, n_classes)
        return x

    @classmethod
    def from_config(cls, config: dict = None) -> "HybridQNN":
        if config is None:
            config = load_config()
        m = config["model"]
        return cls(
            n_qubits=m["n_qubits"],
            n_layers=m["n_layers"],
            n_classes=m["n_classes"],
        )
