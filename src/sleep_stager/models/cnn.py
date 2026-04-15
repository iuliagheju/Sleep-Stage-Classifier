"""1D CNN baseline implemented in PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..utils.progress import TrainingProgressTracker


class EpochDataset(Dataset):
    def __init__(self, signals: np.ndarray, labels: np.ndarray, label_to_idx: Dict[str, int]):
        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor([label_to_idx[label] for label in labels], dtype=torch.long)

    def __len__(self) -> int:
        return self.signals.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.signals[idx], self.labels[idx]


class SimpleCNN(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(slots=True)
class CNNConfig:
    epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"


def train_model(
    signals: np.ndarray,
    labels: np.ndarray,
    label_to_idx: Dict[str, int],
    config: CNNConfig,
) -> Tuple[SimpleCNN, Dict[str, float]]:
    dataset = EpochDataset(signals, labels, label_to_idx)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    num_classes = len(label_to_idx)
    channels = signals.shape[1]
    model = SimpleCNN(channels, num_classes).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    history: Dict[str, float] = {}
    tracker = TrainingProgressTracker(total_steps=config.epochs, label="cnn")
    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_x.size(0)
        epoch_loss = running_loss / len(dataset)
        history[f"epoch_{epoch}_loss"] = epoch_loss
        print(tracker.message(epoch + 1, detail=f"loss={epoch_loss:.4f}"), flush=True)
    return model, history


def predict(model: SimpleCNN, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(signals, dtype=torch.float32)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    return preds, probs
