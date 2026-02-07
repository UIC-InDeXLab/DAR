from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import faiss as faiss_lib


class AnnIndex(ABC):
    @abstractmethod
    def build(self, pts: np.ndarray, metric: str) -> "AnnIndex":
        raise NotImplementedError

    @abstractmethod
    def knn(self, q: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class _FaissBase(AnnIndex):
    index: Optional[faiss_lib.Index] = None
    d: Optional[int] = None
    metric: Optional[str] = None

    def knn(self, q: np.ndarray, k: int) -> np.ndarray:
        q = q.reshape(1, -1)  # FAISS expects 2D array

        # Normalize query for cosine similarity
        if self.metric in ("cosine", "angular"):
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            q = q / norms

        _, indices = self.index.search(q, k + 10)  # return a larger set to filter later

        return indices[0]


class FaissFlatIndex(_FaissBase):
    """Exact flat index (no compression).

    - L2/euclidean: `faiss.IndexFlatL2`
    - cosine/angular: `faiss.IndexFlatIP` with row-normalized vectors
    """

    def build(self, pts: np.ndarray, metric: str) -> "FaissFlatIndex":
        self.data = pts
        self.d = int(pts.shape[1])
        self.metric = metric

        if metric in ("cosine", "angular"):
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts_normalized = pts / norms
            index = faiss_lib.IndexFlatIP(self.d)
            index.add(pts_normalized)
        elif metric in ("euclidean", "l2"):
            index = faiss_lib.IndexFlatL2(self.d)
            index.add(pts)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self.index = index
        return self


class FaissIVFPQIndex(_FaissBase):
    def __init__(
        self,
        nlist: int = 1024,
        m: int = 50,
        nbits: int = 8,
        nprobe: int = 128,  # Increased from 8 to 64 for better recall
    ):
        super().__init__()
        self.nlist = int(nlist)
        self.m = int(m)
        self.nbits = int(nbits)
        self.nprobe = int(nprobe)

    def build(self, pts: np.ndarray, metric: str) -> "FaissIVFPQIndex":
        self.data = pts
        self.d = int(pts.shape[1])
        self.metric = metric
        n = int(pts.shape[0])

        # FAISS PQ training needs at least 2**nbits training points.
        # Also, IVF coarse quantizer needs at least `nlist` training points.
        effective_nlist = min(self.nlist, max(1, n))

        if n <= 1:
            effective_nbits = 1
        else:
            max_nbits = int(np.floor(np.log2(n)))
            effective_nbits = max(1, min(self.nbits, max_nbits))

        # m must divide d for PQ.
        effective_m = min(self.m, self.d)
        if self.d % effective_m != 0:
            for candidate in range(effective_m, 0, -1):
                if self.d % candidate == 0:
                    effective_m = candidate
                    break

        if metric in ("cosine", "angular"):
            # Normalize points for cosine similarity
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            pts_normalized = pts / norms
            quantizer = faiss_lib.IndexFlatIP(self.d)
            index = faiss_lib.IndexIVFPQ(
                quantizer,
                self.d,
                int(effective_nlist),
                int(effective_m),
                int(effective_nbits),
                faiss_lib.METRIC_INNER_PRODUCT,
            )
            index.train(pts_normalized)
            index.add(pts_normalized)
        elif metric in ("euclidean", "l2"):
            quantizer = faiss_lib.IndexFlatL2(self.d)
            index = faiss_lib.IndexIVFPQ(
                quantizer,
                self.d,
                int(effective_nlist),
                int(effective_m),
                int(effective_nbits),
            )
            index.train(pts)
            index.add(pts)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        index.nprobe = self.nprobe
        self.index = index
        return self


class FaissIVFFlatIndex(_FaissBase):
    """IVF over raw vectors (IndexIVFFlat).

    - L2/euclidean: `faiss.IndexIVFFlat` with `METRIC_L2`
    - cosine/angular: `faiss.IndexIVFFlat` with `METRIC_INNER_PRODUCT` and
      row-normalized vectors
    """

    def __init__(self, nlist: int = 128, nprobe: int = 64):
        super().__init__()
        self.nlist = int(nlist)
        self.nprobe = int(nprobe)

    def build(self, pts: np.ndarray, metric: str) -> "FaissIVFFlatIndex":
        self.data = pts
        self.d = int(pts.shape[1])
        self.metric = metric
        n = int(pts.shape[0])

        # Extremely small datasets: avoid IVF training edge cases.
        if n <= 1:
            if metric in ("cosine", "angular"):
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                pts_normalized = pts / norms
                index = faiss_lib.IndexFlatIP(self.d)
                index.add(pts_normalized)
            elif metric in ("euclidean", "l2"):
                index = faiss_lib.IndexFlatL2(self.d)
                index.add(pts)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            self.index = index
            return self

        effective_nlist = min(self.nlist, max(1, n))

        if metric in ("cosine", "angular"):
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts_normalized = pts / norms

            quantizer = faiss_lib.IndexFlatIP(self.d)
            index = faiss_lib.IndexIVFFlat(
                quantizer,
                self.d,
                int(effective_nlist),
                faiss_lib.METRIC_INNER_PRODUCT,
            )
            index.train(pts_normalized)
            index.add(pts_normalized)
        elif metric in ("euclidean", "l2"):
            quantizer = faiss_lib.IndexFlatL2(self.d)
            index = faiss_lib.IndexIVFFlat(
                quantizer,
                self.d,
                int(effective_nlist),
                faiss_lib.METRIC_L2,
            )
            index.train(pts)
            index.add(pts)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        index.nprobe = min(self.nprobe, int(effective_nlist))
        self.index = index
        return self


class FaissPQFlatIndex(_FaissBase):
    """Product Quantization without IVF (flat PQ).

    Uses `faiss.IndexPQ`.
    """

    def __init__(self, m: int = 100, nbits: int = 8):
        super().__init__()
        self.m = int(m)
        self.nbits = int(nbits)

    def build(self, pts: np.ndarray, metric: str) -> "FaissPQFlatIndex":
        self.data = pts
        self.d = int(pts.shape[1])
        self.metric = metric
        n = int(pts.shape[0])

        if n <= 1:
            effective_nbits = 1
        else:
            max_nbits = int(np.floor(np.log2(n)))
            effective_nbits = max(1, min(self.nbits, max_nbits))

        # m must divide d for PQ.
        effective_m = min(self.m, self.d)
        if self.d % effective_m != 0:
            for candidate in range(effective_m, 0, -1):
                if self.d % candidate == 0:
                    effective_m = candidate
                    break

        if metric in ("cosine", "angular"):
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts_normalized = pts / norms

            index = faiss_lib.IndexPQ(
                self.d,
                int(effective_m),
                int(effective_nbits),
                faiss_lib.METRIC_INNER_PRODUCT,
            )
            index.train(pts_normalized)
            index.add(pts_normalized)
        elif metric in ("euclidean", "l2"):
            index = faiss_lib.IndexPQ(
                self.d,
                int(effective_m),
                int(effective_nbits),
                faiss_lib.METRIC_L2,
            )
            index.train(pts)
            index.add(pts)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self.index = index
        return self


class FaissHNSWFlatIndex(_FaissBase):
    """HNSW over raw vectors (HNSWFlat)."""

    def __init__(self, M: int = 32, ef_construction: int = 40, ef_search: int = 32):
        super().__init__()
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)

    def build(self, pts: np.ndarray, metric: str) -> "FaissHNSWFlatIndex":
        self.data = pts
        self.d = int(pts.shape[1])
        self.metric = metric

        if metric in ("cosine", "angular"):
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts_normalized = pts / norms
            index = faiss_lib.IndexHNSWFlat(
                self.d, self.M, faiss_lib.METRIC_INNER_PRODUCT
            )
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            index.add(pts_normalized)
        elif metric in ("euclidean", "l2"):
            index = faiss_lib.IndexHNSWFlat(self.d, self.M)
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            index.add(pts)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self.index = index
        return self


class FaissHNSWPQIndex(_FaissBase):
    """HNSW with PQ-compressed vectors (HNSWPQ)."""

    def __init__(
        self,
        M: int = 64,
        pq_m: int = 100,
        pq_nbits: int = 8,
        ef_construction: int = 40,
        ef_search: int = 64,
    ):
        super().__init__()
        self.M = int(M)
        self.pq_m = int(pq_m)
        self.pq_nbits = int(pq_nbits)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)

    def build(self, pts: np.ndarray, metric: str) -> "FaissHNSWPQIndex":
        self.data = pts
        self.d = int(pts.shape[1])
        self.metric = metric
        n = int(pts.shape[0])

        if n <= 1:
            effective_nbits = 1
        else:
            max_nbits = int(np.floor(np.log2(n)))
            effective_nbits = max(1, min(self.pq_nbits, max_nbits))

        # pq_m must divide d.
        effective_pq_m = min(self.pq_m, self.d)
        if self.d % effective_pq_m != 0:
            for candidate in range(effective_pq_m, 0, -1):
                if self.d % candidate == 0:
                    effective_pq_m = candidate
                    break

        if metric in ("cosine", "angular"):
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            pts_normalized = pts / norms

            index = faiss_lib.IndexHNSWPQ(
                self.d,
                int(effective_pq_m),
                self.M,
                int(effective_nbits),
                faiss_lib.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            if not index.is_trained:
                index.train(pts_normalized)
            index.add(pts_normalized)
        elif metric in ("euclidean", "l2"):
            index = faiss_lib.IndexHNSWPQ(
                self.d,
                int(effective_pq_m),
                self.M,
                int(effective_nbits),
                faiss_lib.METRIC_L2,
            )
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            if not index.is_trained:
                index.train(pts)
            index.add(pts)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self.index = index
        return self


# Backwards-compatible alias
FaissHNSWIndex = FaissHNSWFlatIndex


class FaissLSHIndex(_FaissBase):
    def __init__(
        self, nbits: int = 2048
    ):  # Increased from 256 to 2048 for better recall
        super().__init__()
        self.nbits = int(nbits)

    def build(self, pts: np.ndarray, metric: str) -> "FaissLSHIndex":
        self.data = pts
        self.d = int(pts.shape[1])
        self.metric = metric

        if metric in ("cosine", "angular"):
            # Normalize points for cosine similarity
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            pts_normalized = pts / norms
            index = faiss_lib.IndexLSH(self.d, self.nbits)
            index.add(pts_normalized)
        elif metric in ("euclidean", "l2"):
            index = faiss_lib.IndexLSH(self.d, self.nbits)
            index.add(pts)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self.index = index
        return self


__all__ = [
    "AnnIndex",
    "FaissFlatIndex",
    "FaissPQFlatIndex",
    "FaissIVFPQIndex",
    "FaissIVFFlatIndex",
    "FaissHNSWFlatIndex",
    "FaissHNSWPQIndex",
    "FaissHNSWIndex",
    "FaissLSHIndex",
]
