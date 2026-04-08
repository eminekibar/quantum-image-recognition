"""
Model Testleri
--------------
Calistirma: pytest tests/test_model.py -v
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum_model import HybridQNN, ClassicalEncoder, QuantumLayer, load_config


CONFIG = load_config()


class TestClassicalEncoder:

    def test_output_shape(self):
        n_qubits = CONFIG["model"]["n_qubits"]
        enc = ClassicalEncoder(n_qubits)
        x = torch.randn(4, 1, 28, 28)
        out = enc(x)
        assert out.shape == (4, n_qubits), f"Beklenen (4, {n_qubits}), alindi {out.shape}"

    def test_output_range(self):
        """Tanh cikisi [-1, 1] araliginda olmali."""
        enc = ClassicalEncoder(4)
        x = torch.randn(8, 1, 28, 28)
        out = enc(x)
        assert out.min().item() >= -1.01
        assert out.max().item() <= 1.01


class TestQuantumLayer:

    def test_output_shape(self):
        n_qubits = CONFIG["model"]["n_qubits"]
        n_layers = CONFIG["model"]["n_layers"]
        ql = QuantumLayer(n_qubits, n_layers)
        x = torch.randn(2, n_qubits)
        out = ql(x)
        assert out.shape == (2, n_qubits)

    def test_has_parameters(self):
        ql = QuantumLayer(4, 2)
        params = list(ql.parameters())
        assert len(params) > 0, "Kuantum katmaninin ogrenilir parametresi olmali"


class TestHybridQNN:

    def test_forward_pass(self):
        model = HybridQNN.from_config(CONFIG)
        x = torch.randn(3, 1, 28, 28)
        out = model(x)
        assert out.shape == (3, 10), f"Beklenen (3, 10), alindi {out.shape}"

    def test_output_is_not_nan(self):
        model = HybridQNN.from_config(CONFIG)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert not torch.isnan(out).any(), "Model cikisinda NaN degeri var"

    def test_gradient_flows(self):
        model = HybridQNN.from_config(CONFIG)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # En az bir parametre icin gradyan hesaplanmis olmali
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "Gradyan akisi yok"

    def test_parameter_count(self):
        model = HybridQNN.from_config(CONFIG)
        total = sum(p.numel() for p in model.parameters())
        print(f"\nToplam parametre sayisi: {total:,}")
        assert total > 0
