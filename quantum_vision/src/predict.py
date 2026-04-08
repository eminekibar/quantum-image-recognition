"""
Tek Goruntu Tahmini
--------------------
Kameradan veya dosyadan goruntu alip tahmin uretiyor.

Kullanim:
    python src/predict.py                  # config'deki kamera modunu kullanir
    python src/predict.py --image resim.png
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum_model import HybridQNN, load_config
from camera.base_camera import BaseCamera


PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def load_model(config: dict, device: torch.device) -> HybridQNN:
    model = HybridQNN.from_config(config).to(device)
    ckpt = config["paths"]["checkpoint"]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint bulunamadi: {ckpt}\nOnce 'python src/train.py' calistirin.")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def predict_frame(model: HybridQNN, frame: np.ndarray, device: torch.device):
    """
    Ham kamera frame'ini (BGR veya gri) alip tahmin uretir.
    Dondurulenler: (sinif_no: int, olasiliklar: list[float], sure_ms: float)
    """
    tensor = PREPROCESS(frame).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
    elapsed = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    pred_class = int(torch.argmax(logits, dim=1).item())
    return pred_class, probs, elapsed


def build_camera(config: dict) -> BaseCamera:
    mode = config["camera"]["mode"]
    if mode == "mock":
        from camera.mock_camera import MockCamera
        return MockCamera(config)
    elif mode == "live":
        from camera.live_camera import LiveCamera
        return LiveCamera(config)
    elif mode == "image":
        from camera.image_camera import ImageCamera
        return ImageCamera(config)
    else:
        raise ValueError(f"Bilinmeyen kamera modu: {mode}")


def run_camera_loop(model, config, device):
    cam = build_camera(config)
    print(f"Kamera modu: {config['camera']['mode']} — cikmak icin 'q' tusuna basin")
    try:
        while cam.is_open():
            frame = cam.read_frame()
            if frame is None:
                break

            pred, probs, ms = predict_frame(model, frame, device)
            confidence = max(probs) * 100

            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
            display = cv2.resize(display, (280, 280))
            cv2.putText(display, f"Tahmin: {pred} (%{confidence:.0f})",
                        (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display, f"{ms:.0f}ms",
                        (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Kuantum Goruntu Tanima", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


def predict_single_image(model, image_path: str, device: torch.device):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Goruntu bulunamadi: {image_path}")
    pred, probs, ms = predict_frame(model, frame, device)
    print(f"Goruntu : {image_path}")
    print(f"Tahmin  : {pred}")
    print(f"Guven   : %{max(probs)*100:.1f}")
    print(f"Sure    : {ms:.1f}ms")
    print(f"Tum olasiliklar: {[f'{p*100:.1f}%' for p in probs]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="Tek goruntu dosya yolu")
    args = parser.parse_args()

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)

    if args.image:
        predict_single_image(model, args.image, device)
    else:
        run_camera_loop(model, config, device)


if __name__ == "__main__":
    main()
