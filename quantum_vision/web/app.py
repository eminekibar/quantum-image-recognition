"""
Flask Web Arayuzu
------------------
Tarayicidan canli demo yapilabilmesini saglar.
Kamera frame'lerini base64 olarak stream eder.

Calistirma:
    python web/app.py
    Tarayicida: http://localhost:5000
"""

import base64
import io
import os
import sys

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum_model import HybridQNN, load_config
from src.predict import predict_frame, build_camera

app = Flask(__name__)

_config = None
_model = None
_device = None
_camera = None


def get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_model():
    global _model, _device
    if _model is None:
        config = get_config()
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = HybridQNN.from_config(config).to(_device)
        ckpt = config["paths"]["checkpoint"]
        if os.path.exists(ckpt):
            _model.load_state_dict(torch.load(ckpt, map_location=_device))
            _model.eval()
    return _model, _device


def get_camera():
    global _camera
    if _camera is None or not _camera.is_open():
        _camera = build_camera(get_config())
    return _camera


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body: JSON {"image": "<base64 PNG>"}  veya bos (kameradan frame alir)
    Dondurulenler: {"prediction": int, "confidence": float, "probabilities": list, "time_ms": float}
    """
    model, device = get_model()

    data = request.get_json(silent=True)
    if data and "image" in data:
        img_bytes = base64.b64decode(data["image"])
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    else:
        cam = get_camera()
        frame = cam.read_frame()
        if frame is None:
            return jsonify({"error": "Kameradan frame alinamadi"}), 500

    pred, probs, ms = predict_frame(model, frame, device)
    return jsonify({
        "prediction": pred,
        "confidence": round(max(probs) * 100, 1),
        "probabilities": [round(p * 100, 1) for p in probs],
        "time_ms": round(ms, 1),
    })


@app.route("/frame")
def frame():
    """GET /frame — Kameradan bir frame'i JPEG base64 olarak dondurur."""
    cam = get_camera()
    f = cam.read_frame()
    if f is None:
        return jsonify({"error": "Frame alinamadi"}), 500

    display = cv2.resize(f, (280, 280))
    if display.ndim == 2:
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", display)
    b64 = base64.b64encode(buf).decode()
    return jsonify({"image": b64})


@app.route("/status")
def status():
    config = get_config()
    ckpt = config["paths"]["checkpoint"]
    return jsonify({
        "model_loaded": os.path.exists(ckpt),
        "camera_mode": config["camera"]["mode"],
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    })


if __name__ == "__main__":
    cfg = get_config()
    app.run(
        host=cfg["web"]["host"],
        port=cfg["web"]["port"],
        debug=cfg["web"]["debug"],
    )
