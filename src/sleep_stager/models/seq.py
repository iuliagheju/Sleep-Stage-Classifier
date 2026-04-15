"""Context-aware CNN-BiLSTM model with windowed epochs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..utils.progress import TrainingProgressTracker


class ContextWindowDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray | None,
        label_to_idx: Dict[str, int] | None,
        window_size: int,
    ):
        if window_size % 2 == 0:
            raise ValueError("window_size must be odd")
        self.window_size = window_size
        self.pad = window_size // 2
        self.signals = torch.tensor(
            np.pad(signals, ((self.pad, self.pad), (0, 0), (0, 0)), mode="edge"),
            dtype=torch.float32,
        )
        if labels is None or label_to_idx is None:
            self.labels = None
        else:
            mapped = [label_to_idx[label] for label in labels]
            self.labels = torch.tensor(mapped, dtype=torch.long)

    def __len__(self) -> int:
        return self.signals.shape[0] - 2 * self.pad

    def __getitem__(self, idx: int):
        start = idx
        window = self.signals[start : start + self.window_size]
        if self.labels is None:
            return window
        return window, self.labels[idx]


class CNNEncoder(nn.Module):
    def __init__(self, channels: int, embed_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64 * 16, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ContextCNNBiLSTM(nn.Module):
    def __init__(self, channels: int, num_classes: int, window_size: int, embed_dim: int, hidden_size: int):
        super().__init__()
        self.window_size = window_size
        self.encoder = CNNEncoder(channels, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, window, channels, samples = x.shape
        feats = self.encoder(x.view(batch * window, channels, samples))
        feats = feats.view(batch, window, -1)
        output, _ = self.lstm(feats)
        center = output[:, self.window_size // 2, :]
        return self.head(center)


@dataclass(slots=True)
class SeqConfig:
    epochs: int = 2
    batch_size: int = 16
    lr: float = 1e-3
    device: str = "cpu"
    window_size: int = 5
    embed_dim: int = 64
    hidden_size: int = 64


def train_model(
    signals: np.ndarray,
    labels: np.ndarray,
    label_to_idx: Dict[str, int],
    config: SeqConfig,
) -> Tuple[ContextCNNBiLSTM, Dict[str, float]]:
    dataset = ContextWindowDataset(signals, labels, label_to_idx, config.window_size)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = ContextCNNBiLSTM(
        channels=signals.shape[1],
        num_classes=len(label_to_idx),
        window_size=config.window_size,
        embed_dim=config.embed_dim,
        hidden_size=config.hidden_size,
    ).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()
    history: Dict[str, float] = {}
    tracker = TrainingProgressTracker(total_steps=config.epochs, label="seq")
    for epoch in range(config.epochs):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch_x.size(0)
        epoch_loss = running / len(dataset)
        history[f"epoch_{epoch}_loss"] = epoch_loss
        print(tracker.message(epoch + 1, detail=f"loss={epoch_loss:.4f}"), flush=True)
    return model, history


def predict(model: ContextCNNBiLSTM, signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        window_size = model.window_size
        dataset = ContextWindowDataset(signals, None, None, window_size)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_probs = []
        for batch_x in loader:
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs
