"""
Statik Goruntu Kamerasi
------------------------
Bir klasordeki goruntuleri sirayla dondurur.
Demo ve sunum icin idealdir — gercek kamera gerektirmez.
"""

import os
import cv2
import numpy as np

from camera.base_camera import BaseCamera

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class ImageCamera(BaseCamera):

    def __init__(self, config: dict):
        image_dir = config["camera"].get("image_dir", "assets/test_images/")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Goruntu klasoru bulunamadi: {image_dir}")

        self._paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in EXTENSIONS
        ])
        if not self._paths:
            raise FileNotFoundError(f"{image_dir} klasorunde desteklenen goruntu bulunamadi.")

        self._index = 0

    def read_frame(self) -> np.ndarray:
        if not self.is_open():
            return None
        frame = cv2.imread(self._paths[self._index])
        self._index += 1
        return frame

    def is_open(self) -> bool:
        return self._index < len(self._paths)

    def release(self):
        self._index = len(self._paths)  # is_open() False yapilir
