from __future__ import annotations
 
from pathlib import Path
 
import numpy as np
import rioxarray as rxr
from PIL import Image
 
import torch
from torch.utils.data import Dataset
 
 
class GlacierDataset(Dataset):
    """
    Dataset pour la segmentation de glaciers.
 
    Charge les composites Sentinel-2 (5 bandes : B, G, R, NIR, SWIR16)
    et les masques GLIMS binaires correspondants.
 
    Les augmentations spatiales (flips, rotations 90°) sont appliquées
    de manière synchronisée au composite et au masque.
 
    Parameters
    ----------
    pairs : list of (Path, Path)
        Liste de tuples (chemin_composite.tif, chemin_masque.png).
    mean, std : np.ndarray or None
        Statistiques par bande pour la normalisation (calculées sur le train).
    augment : bool
        Active les augmentations spatiales aléatoires.
    pad_to : int
        Taille cible après zero-padding (doit être un multiple de 16).
    """
 
    def __init__(
        self,
        pairs: list[tuple[Path, Path]],
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        augment: bool = False,
        pad_to: int = 320,
    ):
        self.pairs = pairs
        self.mean = mean
        self.std = std
        self.augment = augment
        self.pad_to = pad_to
 
    def __len__(self):
        return len(self.pairs)
 
    def __getitem__(self, idx):
        tif_path, mask_path = self.pairs[idx]
 
        # ── Composite (5, H, W) float32 ──
        da = rxr.open_rasterio(tif_path)
        img = da.values.astype(np.float32)  # (C, H, W)
        img = np.nan_to_num(img, nan=0.0)
        #Check que la dimension est bel et bien 5, sinon on crée
        # la bande swir pour que ca marche, même si c'est pas propre
        if img.shape[0] == 4:
            swir = np.zeros_like(img[0:1])
            img = np.concatenate([img, swir], axis=0)
 
        # ── Masque (1, H, W) float32 dans [0, 1] ──
        mask = np.array(Image.open(mask_path), dtype=np.float32) / 255.0
        mask = mask[np.newaxis]  # (1, H, W)
 
        # Normalisation par bande
        if self.mean is not None and self.std is not None:
            for b in range(img.shape[0]):
                img[b] = (img[b] - self.mean[b]) / (self.std[b] + 1e-8)
 
        # Padding à pad_to × pad_to
        img = self._pad(img)
        mask = self._pad(mask)
 
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
 
        # Augmentations synchronisées
        if self.augment:
            img, mask = self._augment(img, mask)
 
        return img, mask
 
    # helpers
 
    def _pad(self, arr: np.ndarray) -> np.ndarray:
        """Zero-pad puis crop centré → (C, pad_to, pad_to) garanti."""
        _, h, w = arr.shape
        t = self.pad_to

        # Pad si trop petit
        if h < t or w < t:
            ph = max(0, t - h)
            pw = max(0, t - w)
            arr = np.pad(
                arr,
                ((0, 0), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)),
                mode="constant",
            )
            _, h, w = arr.shape

        # Crop centré si trop grand
        if h > t or w > t:
            y0 = (h - t) // 2
            x0 = (w - t) // 2
            arr = arr[:, y0 : y0 + t, x0 : x0 + t]

        return arr
 
    @staticmethod
    def _augment(img, mask):
        """Flips + rotations 90° appliqués identiquement à l'image et au masque."""
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [-1])
            mask = torch.flip(mask, [-1])
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [-2])
            mask = torch.flip(mask, [-2])
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            img = torch.rot90(img, k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])
        return img, mask
 
 
def compute_band_stats(
    pairs: list[tuple[Path, Path]], n_bands: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Moyenne et écart-type par bande sur les pixels valides (≠ 0).
 
    À calculer uniquement sur le split train pour éviter le data leakage.
    """
    sums = np.zeros(n_bands)
    sq_sums = np.zeros(n_bands)
    counts = np.zeros(n_bands)
 
    for tif_path, _ in pairs:
        arr = rxr.open_rasterio(tif_path).values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        for b in range(n_bands):
            if (b >= len(arr)): continue
            valid = arr[b] != 0
            sums[b] += arr[b][valid].sum()
            sq_sums[b] += (arr[b][valid] ** 2).sum()
            counts[b] += valid.sum()
 
    mean = sums / (counts + 1e-8)
    std = np.sqrt(sq_sums / (counts + 1e-8) - mean**2)
    return mean.astype(np.float32), std.astype(np.float32)
 
 
def discover_pairs(
    composites_root: Path, masks_root: Path
) -> dict[str, list[tuple[Path, Path]]]:
    """
    Apparie chaque composite .tif à son masque _mask.png.
 
    Returns
    -------
    dict
        {region: [(tif_path, mask_path), ...]}
    """
    region_pairs: dict[str, list[tuple[Path, Path]]] = {}
 
    for tif in sorted(composites_root.glob("*/*.tif")):
        region = tif.parent.name
        mask_png = masks_root / region / (tif.stem + "_mask.png")
        if not mask_png.exists():
            continue
        region_pairs.setdefault(region, []).append((tif, mask_png))
 
    for r, pairs in region_pairs.items():
        print(f"  {r:12s}: {len(pairs)} paires")
    return region_pairs