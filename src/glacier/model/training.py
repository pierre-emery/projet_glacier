from __future__ import annotations
 
import numpy as np
import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
 
from glacier.model.lost_functions import iou_score
 
 
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Entraîne le modèle pour une epoch complète.
 
    Returns
    -------
    (loss_moyenne, iou_moyen)
    """
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    n_samples = 0
 
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
 
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
 
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            running_iou += iou_score(preds, masks) * imgs.size(0)
            running_loss += loss.item() * imgs.size(0)
            n_samples += imgs.size(0)
 
    return running_loss / n_samples, running_iou / n_samples
 
 
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Évalue le modèle sur un DataLoader (validation ou test).
 
    Returns
    -------
    (loss_moyenne, iou_moyen)
    """
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    n_samples = 0
 
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
 
        logits = model(imgs)
        loss = criterion(logits, masks)
 
        preds = (torch.sigmoid(logits) > 0.5).float()
        running_iou += iou_score(preds, masks) * imgs.size(0)
        running_loss += loss.item() * imgs.size(0)
        n_samples += imgs.size(0)
 
    return running_loss / n_samples, running_iou / n_samples
 
 
def plot_history(history: dict) -> None:
    """Affiche les courbes d'entraînement (loss, IoU, learning rate)."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
 
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Dice + BCE Loss")
    ax1.legend()
 
    ax2.plot(history["train_iou"], label="Train")
    ax2.plot(history["val_iou"], label="Val")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.set_title("IoU")
    ax2.legend()
 
    ax3.plot(history["lr"])
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("LR Schedule")
 
    plt.tight_layout()
    plt.show()
 
 
def show_predictions(
    model: nn.Module,
    dataset,
    device: torch.device,
    n: int = 4,
) -> None:
    """Affiche n triplets (RGB, masque GLIMS, prédiction U-Net)."""
    model.eval()
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis]
 
    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)
 
    for row, idx in enumerate(indices):
        img, mask = dataset[idx]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits) > 0.5).float().cpu()
 
        # RGB pour affichage (bandes 2=R, 1=G, 0=B)
        rgb = img[[2, 1, 0]].numpy()
        for c in range(3):
            valid = rgb[c][rgb[c] != 0]
            if len(valid) > 0:
                lo, hi = np.percentile(valid, [2, 98])
                rgb[c] = np.clip((rgb[c] - lo) / (hi - lo + 1e-6), 0, 1)
 
        axes[row, 0].imshow(rgb.transpose(1, 2, 0))
        axes[row, 0].set_title("Composite RGB")
        axes[row, 1].imshow(mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Masque GLIMS")
        axes[row, 2].imshow(pred[0, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[row, 2].set_title("Prédiction U-Net")
 
        for ax in axes[row]:
            ax.axis("off")
 
    plt.tight_layout()
    plt.show()