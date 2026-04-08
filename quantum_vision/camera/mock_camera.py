"""
Mock Kamera
------------
Gercek kamera olmadan tam test yapilabilmesi icin
MNIST test setinden rastgele frame uretiyor.

Kullanim:
    cam = MockCamera(config)
    frame = cam.read_frame()   # (28, 28) uint8 numpy dizisi
    label = cam.current_label  # Gercek etiket (test icin)
"""

import random

import numpy as np
import torch
from torchvision import datasets, transforms

from camera.base_camera import BaseCamera


class MockCamera(BaseCamera):

    def __init__(self, config: dict):
        data_dir = config["paths"]["data"]
        transform = transforms.Compose([transforms.ToTensor()])
        self._dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        self._open = True
        self.current_label: int = -1

    def read_frame(self) -> np.ndarray:
        if not self._open:
            return None
        idx = random.randint(0, len(self._dataset) - 1)
        tensor, label = self._dataset[idx]
        self.current_label = int(label)
        # (1, 28, 28) float [0,1] -> (28, 28) uint8 [0, 255]
        frame = (tensor.squeeze().numpy() * 255).astype(np.uint8)
        return frame

    def is_open(self) -> bool:
        return self._open

    def release(self):
        self._open = False

    def get_batch(self, n: int) -> list[tuple[np.ndarray, int]]:
        """n adet (frame, label) cifti dondurur — toplu test icin."""
        batch = []
        for _ in range(n):
            frame = self.read_frame()
            batch.append((frame, self.current_label))
        return batch
