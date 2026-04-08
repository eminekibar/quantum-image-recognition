"""
Tahmin Testleri
----------------
Checkpoint yoksa testler atlanir.
Calistirma: pytest tests/test_predict.py -v
"""

import os
import sys
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum_model import HybridQNN, load_config
from src.predict import predict_frame
from camera.mock_camera import MockCamera


CONFIG = load_config()
CKPT = CONFIG["paths"]["checkpoint"]


@pytest.fixture
def model_and_device():
    if not os.path.exists(CKPT):
        pytest.skip("Checkpoint bulunamadi — once 'python src/train.py' calistirin")
    device = torch.device("cpu")
    model = HybridQNN.from_config(CONFIG).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    return model, device


def test_predict_returns_valid_class(model_and_device):
    model, device = model_and_device
    cam = MockCamera(CONFIG)
    frame = cam.read_frame()
    cam.release()

    pred, probs, ms = predict_frame(model, frame, device)
    assert 0 <= pred <= 9
    assert len(probs) == 10
    assert abs(sum(probs) - 1.0) < 0.01, "Olasiliklar toplamı 1 olmali"
    assert ms > 0


def test_predict_speed(model_and_device):
    """Tahmin 2 saniyeden uzun surmemeli (CPU'da simülatör dahil)."""
    model, device = model_and_device
    cam = MockCamera(CONFIG)
    frame = cam.read_frame()
    cam.release()

    _, _, ms = predict_frame(model, frame, device)
    assert ms < 2000, f"Tahmin cok yavas: {ms:.0f}ms"


def test_batch_accuracy(model_and_device):
    """
    50 ornekten en az %50'si dogru tahmin edilmeli.
    Not: Modelin genel dogrulugu ~%72 olmakla birlikte,
    kucuk rastgele batch'lerde istatistiksel varyans yuksektir.
    Gercek dogruluk icin evaluate.py kullanin.
    """
    model, device = model_and_device
    cam = MockCamera(CONFIG)
    batch = cam.get_batch(50)
    cam.release()

    correct = 0
    for frame, true_label in batch:
        pred, _, _ = predict_frame(model, frame, device)
        if pred == true_label:
            correct += 1

    acc = correct / len(batch)
    print(f"\nBatch dogrulugu: %{acc*100:.1f} ({correct}/{len(batch)})")
    assert acc >= 0.50, f"Dogruluk cok dusuk: %{acc*100:.1f}"
