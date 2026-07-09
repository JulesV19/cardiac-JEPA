"""Construit le cache `.npy` de tous les ECG 100 Hz (une seule fois).

Lit les 21 837 enregistrements via wfdb dans l'ordre du CSV et les empile dans
`cache/ptbxl_100hz.npy` de forme (N, 1000, 12) float32 (~1 Go). Signaux BRUTS
(non normalisés) : la z-norm est faite à la volée dans le Dataset.

Usage :  python -m jepa.build_cache [--data-dir DIR] [--out CACHE.npy]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import wfdb

from .data import CACHE_PATH, DATA_DIR, N_LEADS, N_SAMPLES, load_metadata


def build_cache(data_dir: Path = DATA_DIR, out: Path = CACHE_PATH) -> None:
    df = load_metadata(data_dir)
    n = len(df)
    out.parent.mkdir(parents=True, exist_ok=True)

    # memmap en écriture pour ne pas tout garder en RAM pendant la construction.
    arr = np.lib.format.open_memmap(
        out, mode="w+", dtype=np.float32, shape=(n, N_SAMPLES, N_LEADS))

    t0 = time.time()
    for pos, fname in enumerate(df.filename_lr.to_numpy()):
        sig, _ = wfdb.rdsamp(str(data_dir / fname))
        assert sig.shape == (N_SAMPLES, N_LEADS), f"{fname}: {sig.shape}"
        arr[pos] = sig.astype(np.float32)
        if pos % 1000 == 0:
            print(f"{pos:6d}/{n}  ({time.time()-t0:5.1f}s)", flush=True)
    arr.flush()
    print(f"Cache écrit : {out}  shape={arr.shape}  "
          f"({out.stat().st_size/1e9:.2f} Go)  en {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(CACHE_PATH))
    a = ap.parse_args()
    build_cache(Path(a.data_dir), Path(a.out))
