"""Barre de progression partagée : tqdm si dispo, sinon un repli silencieux minimal.

Évite les logs verbeux par epoch : on affiche une barre unique qui se met à jour en place.
"""
from __future__ import annotations

try:
    from tqdm.auto import tqdm
except Exception:                       # repli si tqdm absent
    class tqdm:                          # type: ignore
        def __init__(self, iterable=None, total=None, **kw):
            self.iterable = iterable if iterable is not None else range(total or 0)

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, *a, **k):
            pass

        def update(self, n=1):
            pass

        def close(self):
            pass

        @staticmethod
        def write(s):
            print(s, flush=True)
