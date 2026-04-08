"""
Model Egitimi
-------------
Calistirma:
    python src/train.py
"""

import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.quantum_model import HybridQNN, load_config


def get_dataloaders(config: dict):
    t = config["training"]
    data_dir = config["paths"]["data"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    torch.manual_seed(t["seed"])
    train_ds = Subset(train_ds, torch.randperm(len(train_ds))[: t["train_samples"]])
    test_ds = Subset(test_ds, torch.randperm(len(test_ds))[: t["test_samples"]])

    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=t["batch_size"], shuffle=False)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def main():
    config = load_config()
    t = config["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    model = HybridQNN.from_config(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = get_dataloaders(config)
    print(f"Egitim ornekleri: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")

    log = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    ckpt_path = config["paths"]["checkpoint"]
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

    for epoch in range(1, t["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)

        log["train_loss"].append(tr_loss)
        log["train_acc"].append(tr_acc)
        log["test_loss"].append(te_loss)
        log["test_acc"].append(te_acc)

        print(f"Epoch {epoch:02d}/{t['epochs']} | "
              f"Egitim: kayip={tr_loss:.4f} dogr=%{tr_acc*100:.1f} | "
              f"Test: kayip={te_loss:.4f} dogr=%{te_acc*100:.1f}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> En iyi model kaydedildi (%{best_acc*100:.1f})")

    log_path = config["paths"]["log"]
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nEgitim tamamlandi. En iyi test dogrulugu: %{best_acc*100:.1f}")
    print(f"Model: {ckpt_path} | Log: {log_path}")


if __name__ == "__main__":
    main()
