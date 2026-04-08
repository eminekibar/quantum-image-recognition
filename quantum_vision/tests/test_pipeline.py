"""
Uctan Uca Entegrasyon Testleri
--------------------------------
Kamera -> On isleme -> Model -> Cikti tam akisini test eder.
Checkpoint yoksa testler atlanir.
Calistirma: pytest tests/test_pipeline.py -v
"""

import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum_model import HybridQNN, load_config
from src.predict import predict_frame, build_camera


CONFIG = load_config()
CKPT = CONFIG["paths"]["checkpoint"]


@pytest.fixture
def pipeline():
    if not os.path.exists(CKPT):
        pytest.skip("Checkpoint bulunamadi — once 'python src/train.py' calistirin")
    device = torch.device("cpu")
    model = HybridQNN.from_config(CONFIG).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    # Mock kamera ile calistir
    config = {**CONFIG, "camera": {**CONFIG["camera"], "mode": "mock"}}
    cam = build_camera(config)
    return model, device, cam


def test_full_pipeline_single_frame(pipeline):
    model, device, cam = pipeline
    frame = cam.read_frame()
    assert frame is not None

    pred, probs, ms = predict_frame(model, frame, device)
    assert 0 <= pred <= 9
    assert len(probs) == 10
    cam.release()


def test_pipeline_10_frames(pipeline):
    model, device, cam = pipeline
    results = []
    for _ in range(10):
        if not cam.is_open():
            break
        frame = cam.read_frame()
        if frame is None:
            break
        pred, probs, ms = predict_frame(model, frame, device)
        results.append((pred, ms))

    cam.release()
    assert len(results) == 10, "10 frame islenmeli"
    for pred, ms in results:
        assert 0 <= pred <= 9
