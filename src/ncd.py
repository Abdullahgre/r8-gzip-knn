# Normalised Compression Distance with gzip, as in Jiang et al. (2023):
#   NCD(x, y) = (C(xy) - min(C(x), C(y))) / max(C(x), C(y))
# where C is the gzip-compressed length and xy is the two texts joined by a space.
import gzip
import os
from multiprocessing import Pool

import numpy as np

from data import ROOT

CACHE = os.path.join(ROOT, "results", "cache")


def clen(text):
    return len(gzip.compress(text.encode("utf-8")))


# each worker process keeps its own copy of the reference texts
_refs, _ref_lens = None, None


def _init(refs, ref_lens):
    global _refs, _ref_lens
    _refs, _ref_lens = refs, ref_lens


def _row(query):
    refs, ref_lens = _refs, _ref_lens
    cq = clen(query)
    row = np.zeros(len(refs), dtype=np.float32)
    for j, r in enumerate(refs):
        cqr = clen(query + " " + r)
        row[j] = (cqr - min(cq, ref_lens[j])) / max(cq, ref_lens[j])
    return row


def ncd_matrix(queries, refs, name=None):
    """Distance from every query to every reference. Saved to results/cache so it is
    only computed once (this is the slow part, a few minutes per matrix)."""
    if name:
        path = os.path.join(CACHE, name + ".npy")
        if os.path.exists(path):
            return np.load(path)
    ref_lens = [clen(r) for r in refs]
    print(f"computing gzip distances: {len(queries)} x {len(refs)} ...", flush=True)
    with Pool(initializer=_init, initargs=(refs, ref_lens)) as pool:
        rows = pool.map(_row, queries, chunksize=20)
    D = np.vstack(rows)
    if name:
        os.makedirs(CACHE, exist_ok=True)
        np.save(path, D)
    return D
