"""
Gercek Webcam / Pi Camera Kamerasi
------------------------------------
USB webcam ve Raspberry Pi Camera Module (CSI) destekler.

Pi Camera kurulumu icin (bir kez yapilir):
  sudo raspi-config -> Interface Options -> Legacy Camera -> Enable -> Reboot
  Veya: /boot/config.txt dosyasina  start_x=1  satirini ekle, reboot yap.
  Kontrol: ls /dev/video*  =>  /dev/video0 gorunmeli
"""

import cv2
import numpy as np

from camera.base_camera import BaseCamera


class LiveCamera(BaseCamera):

    def __init__(self, config: dict, device_index: int = 0):
        # Pi Camera icin once V4L2 backend dene, basarisiz olursa standarda don
        self._cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                "Kamera acilamadi!\n"
                "Pi Camera icin: sudo raspi-config -> Interface Options -> Legacy Camera -> Enable -> Reboot\n"
                "USB webcam icin: kameranin takili oldugunu kontrol edin (ls /dev/video*)\n"
                "Kamera yoksa: config.yaml'da camera.mode: mock yapabilirsiniz."
            )
        w = config["camera"].get("frame_width", 640)
        h = config["camera"].get("frame_height", 480)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Pi icin gecikmeyi azaltir

    def read_frame(self) -> np.ndarray:
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame  # BGR (H, W, 3)

    def is_open(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        self._cap.release()
