"""Pipeline de données PTB-XL pour le pré-entraînement JEPA (100 Hz, 12 dérivations).

- Lecture des signaux via un cache `.npy` construit par `build_cache.py` (rapide),
  avec repli sur `wfdb` si le cache est absent.
- Normalisation z-norm par dérivation et par enregistrement, à la volée.
- Splits basés sur `strat_fold` : pretrain 1-8, val (monitoring collapse) 9, test 10 réservé.

Aucune augmentation ici (I-JEPA repose sur le masquage). Le masquage est fait côté
collator (`masking.py`), pas ici : le Dataset renvoie juste le signal normalisé.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Chemins surchargeables par variables d'environnement (Colab, Drive, etc.).
_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get(
    "PTBXL_DATA_DIR",
    _ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1"))
CACHE_PATH = Path(os.environ.get("PTBXL_CACHE", _ROOT / "cache" / "ptbxl_100hz.npy"))

N_SAMPLES = 1000      # 10 s à 100 Hz
N_LEADS = 12
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_metadata(data_dir: Path | None = None) -> pd.DataFrame:
    """Charge ptbxl_database.csv avec scp_codes parsés et superclasses agrégées.

    L'ordre des lignes définit l'index (0..N-1) utilisé par le cache `.npy`.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    df = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    scp_diag = scp[scp.diagnostic == 1]
    df["superclass"] = df.scp_codes.apply(
        lambda d: sorted({scp_diag.loc[k].diagnostic_class
                          for k in d if k in scp_diag.index}))
    return df


def zscore_per_lead(sig: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-norm par dérivation sur les 10 s. `sig` de forme (N_SAMPLES, N_LEADS).

    eps protège les dérivations plates (std≈0), fréquentes dans PTB-XL.
    """
    mean = sig.mean(axis=0, keepdims=True)
    std = sig.std(axis=0, keepdims=True)
    return (sig - mean) / (std + eps)


class PTBXLDataset(Dataset):
    """ECG normalisés pour un sous-ensemble de folds.

    Renvoie un tenseur float32 (N_SAMPLES, N_LEADS). Le masquage n'est PAS fait ici.
    """

    def __init__(self, split: str = "pretrain", data_dir: Path | None = None,
                 cache_path: Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        cache_path = Path(cache_path) if cache_path else CACHE_PATH

        df = load_metadata(self.data_dir)
        if split == "pretrain":
            mask = df.strat_fold.isin(range(1, 9))
        elif split == "val":
            mask = df.strat_fold == 9
        elif split == "test":
            mask = df.strat_fold == 10
        elif split == "all":
            mask = np.ones(len(df), dtype=bool)
        else:
            raise ValueError(f"split inconnu: {split}")

        # Position 0-based dans l'ordre du CSV = index dans le cache .npy.
        self.positions = np.where(mask.to_numpy())[0]
        self.df = df
        self._filenames = df.filename_lr.to_numpy()

        # Cache mémoire (memmap = lazy, pas de copie RAM immédiate).
        self._cache = None
        if cache_path.exists():
            self._cache = np.load(cache_path, mmap_mode="r")
            if self._cache.shape[0] != len(df):
                raise RuntimeError(
                    f"Cache {cache_path} ({self._cache.shape[0]} lignes) désynchronisé "
                    f"du CSV ({len(df)}). Reconstruis-le avec build_cache.py.")
        else:
            import wfdb  # repli lent, uniquement si pas de cache
            self._wfdb = wfdb

    def __len__(self) -> int:
        return len(self.positions)

    def _read_raw(self, pos: int) -> np.ndarray:
        if self._cache is not None:
            return np.asarray(self._cache[pos], dtype=np.float32)
        sig, _ = self._wfdb.rdsamp(str(self.data_dir / self._filenames[pos]))
        return sig.astype(np.float32)

    def __getitem__(self, i: int) -> torch.Tensor:
        pos = int(self.positions[i])
        sig = self._read_raw(pos)
        sig = zscore_per_lead(sig)
        return torch.from_numpy(np.ascontiguousarray(sig, dtype=np.float32))
