"""
Model Degerlendirme & Metrikler
--------------------------------
Egitim sonrasi model performansini analiz eder.
Karisiklik matrisi ve sinif bazli dogruluk raporu uretir.

Calistirma:
    python src/evaluate.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum_model import HybridQNN, load_config


def load_model(config: dict, device: torch.device) -> HybridQNN:
    model = HybridQNN.from_config(config).to(device)
    ckpt = config["paths"]["checkpoint"]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint bulunamadi: {ckpt}\nOnce 'python src/train.py' calistirin.")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


def get_predictions(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def confusion_matrix(preds, labels, n_classes=10):
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for p, l in zip(preds, labels):
        matrix[l][p] += 1
    return matrix


def plot_confusion_matrix(matrix, save_path="assets/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gercek")
    ax.set_title("Karisiklik Matrisi — Hibrit QNN")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                    color="white" if matrix[i][j] > matrix.max() / 2 else "black")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Karisiklik matrisi kaydedildi: {save_path}")


def plot_training_log(log_path: str, save_path="assets/training_curves.png"):
    if not os.path.exists(log_path):
        print(f"Log dosyasi bulunamadi: {log_path}")
        return
    with open(log_path) as f:
        log = json.load(f)

    epochs = range(1, len(log["train_acc"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, log["train_acc"], label="Egitim")
    ax1.plot(epochs, log["test_acc"], label="Test")
    ax1.set_title("Dogruluk")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Dogruluk")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, log["train_loss"], label="Egitim")
    ax2.plot(epochs, log["test_loss"], label="Test")
    ax2.set_title("Kayip")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Kayip")
    ax2.legend()
    ax2.grid(True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Egitim egrileri kaydedildi: {save_path}")


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)

    t = config["training"]
    data_dir = config["paths"]["data"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    test_ds = Subset(test_ds, list(range(t["test_samples"])))
    loader = DataLoader(test_ds, batch_size=64)

    preds, labels = get_predictions(model, loader, device)
    acc = (preds == labels).mean()
    print(f"Test dogrulugu: %{acc*100:.2f} ({(preds==labels).sum()}/{len(labels)})")

    # Sinif bazli dogruluk
    print("\nSinif bazli dogruluk:")
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            c_acc = (preds[mask] == labels[mask]).mean()
            print(f"  Rakam {c}: %{c_acc*100:.1f} ({mask.sum()} ornek)")

    cm = confusion_matrix(preds, labels)
    plot_confusion_matrix(cm)
    plot_training_log(config["paths"]["log"])


if __name__ == "__main__":
    main()
