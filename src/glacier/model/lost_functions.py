from __future__ import annotations
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class DiceBCELoss(nn.Module):
    """
    Combinaison de Binary Cross-Entropy et Dice Loss.
 
    La BCE assure une convergence stable, tandis que la Dice Loss
    gère le déséquilibre de classes (souvent plus de fond que de glacier).
 
    Parameters
    ----------
    bce_weight, dice_weight : float
        Poids relatifs des deux termes.
    smooth : float
        Terme de lissage pour éviter la division par zéro dans le Dice.
    """
 
    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE (applique sigmoid en interne)
        bce = self.bce(logits, targets)
 
        # Dice
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)
 
        return self.bce_w * bce + self.dice_w * dice.mean()
 
class FocalLoss(nn.Module):
    """
    Focal Loss pour la segmentation binaire (Lin et al., 2017).
 
    Réduit le poids des pixels faciles (fond homogène) et concentre
    l'apprentissage sur les pixels difficiles (bordures de glaciers,
    zones mixtes neige/glace).
 
    Parameters
    ----------
    alpha : float
        Poids de la classe positive (glacier). Augmenter si les glaciers
        sont très peu représentés (ex : 0.75 pour données très déséquilibrées).
    gamma : float
        Facteur de focalisation. gamma=0 → BCE classique.
        gamma=2 est la valeur standard recommandée par les auteurs.
    """
 
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        # p_t : probabilité de la classe correcte pour chaque pixel
        p_t = probs * targets + (1 - probs) * (1 - targets)
        # alpha_t : poids de classe selon la cible
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()

class DiceFocalLoss(nn.Module):
    """
    Combinaison Dice Loss + Focal Loss.
    Parameters
    ----------
    dice_weight, focal_weight : float
        Poids relatifs des deux termes.
    smooth : float
        Lissage du Dice.
    alpha, gamma : float
        Paramètres de la Focal Loss (voir FocalLoss).
    """
 
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        smooth: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.dice_w = dice_weight
        self.focal_w = focal_weight
        self.smooth = smooth
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
 
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Dice
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (1.0 - (2.0 * inter + self.smooth) / (union + self.smooth)).mean()
 
        # Focal
        focal = self.focal(logits, targets)
 
        return self.dice_w * dice + self.focal_w * focal


def iou_score(
    preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6
) -> float:
    """
    IoU (Intersection over Union) moyen sur un batch.
 
    Parameters
    ----------
    preds : Tensor (B, 1, H, W)
        Prédictions binaires (0 ou 1).
    targets : Tensor (B, 1, H, W)
        Masques de référence (0 ou 1).
 
    Returns
    -------
    float
        IoU moyen sur le batch.
    """
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    iou = (inter + smooth) / (union + smooth)
    return iou.mean().item()