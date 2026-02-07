import numpy as np


def brute_force(data: np.ndarray, q: np.ndarray, k: int, metric: str) -> np.ndarray:
    if metric == "l2":
        d = np.linalg.norm(data - q, axis=1)
        idx = int(np.argpartition(d, k - 1)[k - 1])
        return idx

    if metric == "cosine":
        qn = np.linalg.norm(q)
        if qn == 0:
            raise ValueError("q has zero norm (cosine undefined).")
        dn = np.linalg.norm(data, axis=1)
        dn = np.where(dn == 0, 1.0, dn)
        sims = (data @ q) / (dn * qn)
        dist = 1.0 - sims  # smaller is closer
        idx = int(np.argpartition(dist, k - 1)[k - 1])
        return idx
