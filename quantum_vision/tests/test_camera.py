"""
Kamera Testleri
----------------
Calistirma: pytest tests/test_camera.py -v
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum_model import load_config
from camera.mock_camera import MockCamera


CONFIG = load_config()


class TestMockCamera:

    def test_read_frame_shape(self):
        cam = MockCamera(CONFIG)
        frame = cam.read_frame()
        assert frame.shape == (28, 28), f"Beklenen (28, 28), alindi {frame.shape}"
        cam.release()

    def test_frame_dtype(self):
        cam = MockCamera(CONFIG)
        frame = cam.read_frame()
        assert frame.dtype == np.uint8, f"Beklenen uint8, alindi {frame.dtype}"
        cam.release()

    def test_frame_value_range(self):
        cam = MockCamera(CONFIG)
        frame = cam.read_frame()
        assert frame.min() >= 0
        assert frame.max() <= 255
        cam.release()

    def test_is_open(self):
        cam = MockCamera(CONFIG)
        assert cam.is_open() is True
        cam.release()
        assert cam.is_open() is False

    def test_release_returns_none(self):
        cam = MockCamera(CONFIG)
        cam.release()
        frame = cam.read_frame()
        assert frame is None

    def test_label_is_valid(self):
        cam = MockCamera(CONFIG)
        cam.read_frame()
        assert 0 <= cam.current_label <= 9
        cam.release()

    def test_context_manager(self):
        with MockCamera(CONFIG) as cam:
            frame = cam.read_frame()
            assert frame is not None
        assert cam.is_open() is False

    def test_batch(self):
        cam = MockCamera(CONFIG)
        batch = cam.get_batch(5)
        assert len(batch) == 5
        for frame, label in batch:
            assert frame.shape == (28, 28)
            assert 0 <= label <= 9
        cam.release()
