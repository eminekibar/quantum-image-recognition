"""
Soyut Kamera Sinifi
--------------------
Tum kamera implementasyonlari bu siniftan turemelidir.
Bu sayede kamera modunu degistirmek icin sadece config.yaml guncellenir.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseCamera(ABC):

    @abstractmethod
    def read_frame(self) -> np.ndarray:
        """
        Bir sonraki frame'i dondurur.
        Cikti: (H, W) gri veya (H, W, 3) BGR numpy dizisi.
        Goruntu yoksa None dondurur.
        """

    @abstractmethod
    def is_open(self) -> bool:
        """Kamera hala aktif mi?"""

    @abstractmethod
    def release(self):
        """Kaynagi serbest birak."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()
