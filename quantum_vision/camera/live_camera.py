"""
Gercek Webcam Kamerasi
-----------------------
OpenCV ile bilgisayarin kamerasina baglanir.
Gerci kamera bagli degilse hata mesaji verir.
"""

import cv2
import numpy as np

from camera.base_camera import BaseCamera


class LiveCamera(BaseCamera):

    def __init__(self, config: dict, device_index: int = 0):
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Kamera acilamadi. Kamera bagli mi? "
                "Mock mod icin config.yaml'da camera.mode: mock yapabilirsiniz."
            )
        w = config["camera"].get("frame_width", 640)
        h = config["camera"].get("frame_height", 480)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def read_frame(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame  # BGR (H, W, 3)

    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._cap.release()
