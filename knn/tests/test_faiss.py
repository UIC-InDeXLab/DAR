import numpy as np
import pytest

try:
    # When running from repo root
    from knn.methods.faiss import (
        FaissFlatIndex,
        FaissPQFlatIndex,
        FaissIVFPQIndex,
        FaissIVFFlatIndex,
        FaissHNSWFlatIndex,
        FaissHNSWPQIndex,
        FaissHNSWIndex,
        FaissLSHIndex,
    )
except Exception:  # pragma: no cover
    # When running from within knn/
    from methods.faiss import (
        FaissFlatIndex,
        FaissPQFlatIndex,
        FaissIVFPQIndex,
        FaissIVFFlatIndex,
        FaissHNSWFlatIndex,
        FaissHNSWPQIndex,
        FaissHNSWIndex,
        FaissLSHIndex,
    )


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def _make_data(n: int = 256, d: int = 8, seed: int = 0) -> np.ndarray:
    return _rng(seed).normal(size=(n, d)).astype(np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize rows to unit length for cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _make_query(d: int, seed: int) -> np.ndarray:
    q = _rng(seed).normal(size=(d,)).astype(np.float32)
    if not np.any(q):
        q[0] = 1.0
    return q


def _true_topk_l2(data: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Compute true top-k nearest neighbors using L2 distance."""
    distances = np.linalg.norm(data - query, axis=1)
    return np.argsort(distances)[:k]


def _true_topk_cosine(data: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Compute true top-k nearest neighbors using cosine similarity."""
    data_norm = _normalize_rows(data)
    query_norm = query / np.linalg.norm(query)
    scores = data_norm @ query_norm
    return np.argsort(-scores)[:k]


# ========== FaissFlatIndex Tests ==========


def test_flat_index_l2_basic():
    """Test FaissFlatIndex with L2 metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    index = FaissFlatIndex().build(data, metric="l2")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert result.dtype == np.int64 or result.dtype == np.int32
    assert all(0 <= idx < 100 for idx in result)


def test_flat_index_euclidean_basic():
    """Test FaissFlatIndex with euclidean metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    index = FaissFlatIndex().build(data, metric="euclidean")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 100 for idx in result)


def test_flat_index_cosine_basic():
    """Test FaissFlatIndex with cosine metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    index = FaissFlatIndex().build(data, metric="cosine")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 100 for idx in result)


def test_flat_index_angular_basic():
    """Test FaissFlatIndex with angular metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    index = FaissFlatIndex().build(data, metric="angular")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 100 for idx in result)


def test_flat_index_l2_correctness():
    """Test FaissFlatIndex returns k+10 neighbors with kth present for L2 metric."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    k = 5
    index = FaissFlatIndex().build(data, metric="l2")
    result = index.knn(query, k=k)
    
    assert len(result) == k + 10
    
    # Check that all true top-k neighbors are present in the result
    expected = _true_topk_l2(data, query, k=k)
    for neighbor in expected:
        assert neighbor in result


def test_flat_index_cosine_correctness():
    """Test FaissFlatIndex returns k+10 neighbors with kth present for cosine metric."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    k = 5
    index = FaissFlatIndex().build(data, metric="cosine")
    result = index.knn(query, k=k)
    
    assert len(result) == k + 10
    
    # Check that all true top-k neighbors are present in the result
    expected = _true_topk_cosine(data, query, k=k)
    for neighbor in expected:
        assert neighbor in result


def test_flat_index_single_neighbor():
    """Test FaissFlatIndex with k=1."""
    np.random.seed(42)
    data = _make_data(n=50, d=8, seed=1)
    query = _make_query(d=8, seed=2)
    
    k = 1
    index = FaissFlatIndex().build(data, metric="l2")
    result = index.knn(query, k=k)
    
    assert len(result) == k + 10
    expected = _true_topk_l2(data, query, k=k)
    assert expected[0] in result


def test_flat_index_all_neighbors():
    """Test FaissFlatIndex with k=n (all neighbors)."""
    np.random.seed(42)
    n = 30
    data = _make_data(n=n, d=8, seed=1)
    query = _make_query(d=8, seed=2)
    
    index = FaissFlatIndex().build(data, metric="l2")
    result = index.knn(query, k=n)
    
    # FAISS pads with -1 when k+10 > n, so we get k+10 results but some may be -1
    assert len(result) == n + 10
    # Filter out -1 (padding) values
    valid_indices = result[result != -1]
    assert len(valid_indices) == n
    assert len(set(valid_indices)) == n  # All unique


def test_flat_index_invalid_metric():
    """Test FaissFlatIndex raises error for invalid metric."""
    np.random.seed(42)
    data = _make_data(n=50, d=8, seed=1)
    
    with pytest.raises(ValueError, match="Unsupported metric"):
        FaissFlatIndex().build(data, metric="invalid_metric")


def test_flat_index_stores_metric():
    """Test FaissFlatIndex stores the metric correctly."""
    np.random.seed(42)
    data = _make_data(n=50, d=8, seed=1)
    
    index = FaissFlatIndex().build(data, metric="cosine")
    assert index.metric == "cosine"
    
    index = FaissFlatIndex().build(data, metric="l2")
    assert index.metric == "l2"


# ========== FaissIVFPQIndex Tests ==========


def test_ivfpq_index_l2_basic():
    """Test FaissIVFPQIndex with L2 metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)
    
    index = FaissIVFPQIndex(nlist=10, m=8, nbits=8, nprobe=3).build(data, metric="l2")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 500 for idx in result)


def test_ivfpq_index_cosine_basic():
    """Test FaissIVFPQIndex with cosine metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)
    
    index = FaissIVFPQIndex(nlist=10, m=8, nbits=8, nprobe=3).build(data, metric="cosine")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 500 for idx in result)


def test_ivfpq_index_cosine_uses_inner_product_metric():
    """Regression test: cosine/"angular" should use IP metric in FAISS."""
    import faiss as faiss_lib

    np.random.seed(42)
    data = _make_data(n=200, d=16, seed=1)

    index = FaissIVFPQIndex(nlist=10, m=8, nbits=8, nprobe=3).build(
        data, metric="cosine"
    )
    assert index.index.metric_type == faiss_lib.METRIC_INNER_PRODUCT


def test_ivfpq_index_euclidean_metric():
    """Test FaissIVFPQIndex with euclidean metric."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)
    
    index = FaissIVFPQIndex(nlist=10, m=8, nbits=8, nprobe=3).build(data, metric="euclidean")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 500 for idx in result)


def test_ivfpq_index_angular_metric():
    """Test FaissIVFPQIndex with angular metric."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)
    
    index = FaissIVFPQIndex(nlist=10, m=8, nbits=8, nprobe=3).build(data, metric="angular")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 500 for idx in result)


def test_ivfpq_index_small_dataset():
    """Test FaissIVFPQIndex handles small datasets correctly."""
    np.random.seed(42)
    data = _make_data(n=20, d=8, seed=1)
    query = _make_query(d=8, seed=2)
    
    # Should handle small n gracefully
    index = FaissIVFPQIndex(nlist=128, m=4, nbits=8, nprobe=3).build(data, metric="l2")
    result = index.knn(query, k=5)
    
    # Since dataset is small, we get k+10 results (may include -1 padding)
    assert len(result) == 15
    # Filter out invalid indices
    valid_indices = result[(result >= 0) & (result < 20)]
    assert len(valid_indices) > 0  # At least some valid results


def test_ivfpq_index_invalid_metric():
    """Test FaissIVFPQIndex raises error for invalid metric."""
    np.random.seed(42)
    data = _make_data(n=100, d=16, seed=1)
    
    with pytest.raises(ValueError, match="Unsupported metric"):
        FaissIVFPQIndex().build(data, metric="invalid_metric")


def test_ivfpq_index_stores_metric():
    """Test FaissIVFPQIndex stores the metric correctly."""
    np.random.seed(42)
    data = _make_data(n=200, d=16, seed=1)
    
    index = FaissIVFPQIndex().build(data, metric="cosine")
    assert index.metric == "cosine"
    
    index = FaissIVFPQIndex().build(data, metric="l2")
    assert index.metric == "l2"


def test_ivfpq_index_parameters():
    """Test FaissIVFPQIndex initializes with correct parameters."""
    index = FaissIVFPQIndex(nlist=64, m=16, nbits=4, nprobe=5)
    
    assert index.nlist == 64
    assert index.m == 16
    assert index.nbits == 4
    assert index.nprobe == 5


# ========== FaissIVFFlatIndex Tests ==========


def test_ivf_index_l2_basic():
    """Test FaissIVFFlatIndex with L2 metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)

    index = FaissIVFFlatIndex(nlist=10, nprobe=3).build(data, metric="l2")
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 500 for idx in result)


def test_ivf_index_cosine_basic():
    """Test FaissIVFFlatIndex with cosine metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)

    index = FaissIVFFlatIndex(nlist=10, nprobe=3).build(data, metric="cosine")
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 500 for idx in result)


def test_ivf_index_euclidean_metric():
    """Test FaissIVFFlatIndex with euclidean metric."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)

    index = FaissIVFFlatIndex(nlist=10, nprobe=3).build(data, metric="euclidean")
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 500 for idx in result)


def test_ivf_index_angular_metric():
    """Test FaissIVFFlatIndex with angular metric."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)

    index = FaissIVFFlatIndex(nlist=10, nprobe=3).build(data, metric="angular")
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 500 for idx in result)


def test_ivf_index_small_dataset():
    """Test FaissIVFFlatIndex handles small datasets correctly."""
    np.random.seed(42)
    data = _make_data(n=20, d=8, seed=1)
    query = _make_query(d=8, seed=2)

    index = FaissIVFFlatIndex(nlist=128, nprobe=3).build(data, metric="l2")
    result = index.knn(query, k=5)

    assert len(result) == 15
    valid_indices = result[(result >= 0) & (result < 20)]
    assert len(valid_indices) > 0


def test_ivf_index_invalid_metric():
    """Test FaissIVFFlatIndex raises error for invalid metric."""
    np.random.seed(42)
    data = _make_data(n=100, d=16, seed=1)

    with pytest.raises(ValueError, match="Unsupported metric"):
        FaissIVFFlatIndex().build(data, metric="invalid_metric")


def test_ivf_index_stores_metric():
    """Test FaissIVFFlatIndex stores the metric correctly."""
    np.random.seed(42)
    data = _make_data(n=200, d=16, seed=1)

    index = FaissIVFFlatIndex().build(data, metric="cosine")
    assert index.metric == "cosine"

    index = FaissIVFFlatIndex().build(data, metric="l2")
    assert index.metric == "l2"


def test_ivf_index_parameters():
    """Test FaissIVFFlatIndex initializes with correct parameters."""
    index = FaissIVFFlatIndex(nlist=64, nprobe=5)

    assert index.nlist == 64
    assert index.nprobe == 5


# ========== FaissPQFlatIndex Tests ==========


def test_pqflat_index_l2_basic():
    """Test FaissPQFlatIndex with L2 metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)

    index = FaissPQFlatIndex(m=8, nbits=4).build(data, metric="l2")
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 500 for idx in result if idx != -1)


def test_pqflat_index_cosine_basic():
    """Test FaissPQFlatIndex with cosine metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=500, d=16, seed=1)
    query = _make_query(d=16, seed=2)

    index = FaissPQFlatIndex(m=8, nbits=4).build(data, metric="cosine")
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 500 for idx in result if idx != -1)


def test_pqflat_index_invalid_metric():
    """Test FaissPQFlatIndex raises error for invalid metric."""
    np.random.seed(42)
    data = _make_data(n=100, d=16, seed=1)

    with pytest.raises(ValueError, match="Unsupported metric"):
        FaissPQFlatIndex().build(data, metric="invalid_metric")


def test_pqflat_index_stores_metric():
    """Test FaissPQFlatIndex stores the metric correctly."""
    np.random.seed(42)
    data = _make_data(n=200, d=16, seed=1)

    index = FaissPQFlatIndex().build(data, metric="cosine")
    assert index.metric == "cosine"

    index = FaissPQFlatIndex().build(data, metric="l2")
    assert index.metric == "l2"


# ========== FaissHNSWIndex Tests ==========


def test_hnsw_index_l2_basic():
    """Test FaissHNSWIndex with L2 metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=200, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    index = FaissHNSWIndex(M=16, ef_construction=40, ef_search=16).build(data, metric="l2")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 200 for idx in result)


def test_hnsw_index_cosine_basic():
    """Test FaissHNSWIndex with cosine metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=200, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    index = FaissHNSWIndex(M=16, ef_construction=40, ef_search=16).build(data, metric="cosine")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 200 for idx in result)


def test_hnsw_index_euclidean_metric():
    """Test FaissHNSWIndex with euclidean metric."""
    np.random.seed(42)
    data = _make_data(n=200, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    index = FaissHNSWIndex(M=16, ef_construction=40, ef_search=16).build(data, metric="euclidean")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 200 for idx in result)


def test_hnsw_index_angular_metric():
    """Test FaissHNSWIndex with angular metric."""
    np.random.seed(42)
    data = _make_data(n=200, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    index = FaissHNSWIndex(M=16, ef_construction=40, ef_search=16).build(data, metric="angular")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 200 for idx in result)


def test_hnsw_index_invalid_metric():
    """Test FaissHNSWIndex raises error for invalid metric."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    
    with pytest.raises(ValueError, match="Unsupported metric"):
        FaissHNSWIndex().build(data, metric="invalid_metric")


def test_hnsw_index_stores_metric():
    """Test FaissHNSWIndex stores the metric correctly."""
    np.random.seed(42)
    data = _make_data(n=200, d=10, seed=1)
    
    index = FaissHNSWIndex().build(data, metric="cosine")
    assert index.metric == "cosine"
    
    index = FaissHNSWIndex().build(data, metric="l2")
    assert index.metric == "l2"


def test_hnsw_index_parameters():
    """Test FaissHNSWIndex initializes with correct parameters."""
    index = FaissHNSWIndex(M=24, ef_construction=50, ef_search=20)
    
    assert index.M == 24
    assert index.ef_construction == 50
    assert index.ef_search == 20


# ========== FaissHNSWFlatIndex Tests ==========


def test_hnswflat_index_l2_basic():
    """Test FaissHNSWFlatIndex with L2 metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=200, d=10, seed=1)
    query = _make_query(d=10, seed=2)

    index = FaissHNSWFlatIndex(M=16, ef_construction=40, ef_search=16).build(
        data, metric="l2"
    )
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 200 for idx in result)


def test_hnswflat_index_cosine_basic():
    """Test FaissHNSWFlatIndex with cosine metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=200, d=10, seed=1)
    query = _make_query(d=10, seed=2)

    index = FaissHNSWFlatIndex(M=16, ef_construction=40, ef_search=16).build(
        data, metric="cosine"
    )
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 200 for idx in result)


# ========== FaissHNSWPQIndex Tests ==========


def test_hnswpq_index_l2_basic():
    """Test FaissHNSWPQIndex with L2 metric on basic data."""
    np.random.seed(42)
    # Use d that is NOT divisible by HNSW M, to ensure we are not accidentally
    # passing HNSW M as PQ subquantizers.
    data = _make_data(n=500, d=10, seed=1)
    query = _make_query(d=10, seed=2)

    index = FaissHNSWPQIndex(
        M=16, pq_m=8, pq_nbits=4, ef_construction=40, ef_search=16
    ).build(data, metric="l2")
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 500 for idx in result if idx != -1)


def test_hnswpq_index_cosine_basic():
    """Test FaissHNSWPQIndex with cosine metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=500, d=10, seed=1)
    query = _make_query(d=10, seed=2)

    index = FaissHNSWPQIndex(
        M=16, pq_m=8, pq_nbits=4, ef_construction=40, ef_search=16
    ).build(data, metric="cosine")
    result = index.knn(query, k=5)

    assert len(result) == 15
    assert all(0 <= idx < 500 for idx in result if idx != -1)


def test_hnswpq_index_invalid_metric():
    """Test FaissHNSWPQIndex raises error for invalid metric."""
    np.random.seed(42)
    data = _make_data(n=100, d=16, seed=1)

    with pytest.raises(ValueError, match="Unsupported metric"):
        FaissHNSWPQIndex().build(data, metric="invalid_metric")


# ========== FaissLSHIndex Tests ==========


def test_lsh_index_l2_basic():
    """Test FaissLSHIndex with L2 metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=200, d=32, seed=1)
    query = _make_query(d=32, seed=2)
    
    index = FaissLSHIndex(nbits=128).build(data, metric="l2")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 200 for idx in result)


def test_lsh_index_cosine_basic():
    """Test FaissLSHIndex with cosine metric on basic data."""
    np.random.seed(42)
    data = _make_data(n=200, d=32, seed=1)
    query = _make_query(d=32, seed=2)
    
    index = FaissLSHIndex(nbits=128).build(data, metric="cosine")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 200 for idx in result)


def test_lsh_index_euclidean_metric():
    """Test FaissLSHIndex with euclidean metric."""
    np.random.seed(42)
    data = _make_data(n=200, d=32, seed=1)
    query = _make_query(d=32, seed=2)
    
    index = FaissLSHIndex(nbits=128).build(data, metric="euclidean")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 200 for idx in result)


def test_lsh_index_angular_metric():
    """Test FaissLSHIndex with angular metric."""
    np.random.seed(42)
    data = _make_data(n=200, d=32, seed=1)
    query = _make_query(d=32, seed=2)
    
    index = FaissLSHIndex(nbits=128).build(data, metric="angular")
    result = index.knn(query, k=5)
    
    assert len(result) == 15  # k + 10
    assert all(0 <= idx < 200 for idx in result)


def test_lsh_index_invalid_metric():
    """Test FaissLSHIndex raises error for invalid metric."""
    np.random.seed(42)
    data = _make_data(n=100, d=32, seed=1)
    
    with pytest.raises(ValueError, match="Unsupported metric"):
        FaissLSHIndex().build(data, metric="invalid_metric")


def test_lsh_index_stores_metric():
    """Test FaissLSHIndex stores the metric correctly."""
    np.random.seed(42)
    data = _make_data(n=200, d=32, seed=1)
    
    index = FaissLSHIndex().build(data, metric="cosine")
    assert index.metric == "cosine"
    
    index = FaissLSHIndex().build(data, metric="l2")
    assert index.metric == "l2"


def test_lsh_index_parameters():
    """Test FaissLSHIndex initializes with correct parameters."""
    index = FaissLSHIndex(nbits=512)
    
    assert index.nbits == 512


# ========== Cross-Index Comparison Tests ==========


def test_flat_vs_hnsw_l2():
    """Test FaissFlatIndex and FaissHNSWIndex give similar results for L2."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    flat_index = FaissFlatIndex().build(data, metric="l2")
    hnsw_index = FaissHNSWIndex(M=32, ef_construction=100, ef_search=50).build(data, metric="l2")
    
    flat_result = flat_index.knn(query, k=5)
    hnsw_result = hnsw_index.knn(query, k=5)
    
    # HNSW is approximate, so check overlap (at least 3 out of 5 should match)
    overlap = len(set(flat_result) & set(hnsw_result))
    assert overlap >= 3, f"Expected at least 3 matches, got {overlap}"


def test_flat_vs_hnsw_cosine():
    """Test FaissFlatIndex and FaissHNSWIndex give similar results for cosine."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    flat_index = FaissFlatIndex().build(data, metric="cosine")
    hnsw_index = FaissHNSWIndex(M=32, ef_construction=100, ef_search=50).build(data, metric="cosine")
    
    flat_result = flat_index.knn(query, k=5)
    hnsw_result = hnsw_index.knn(query, k=5)
    
    # HNSW is approximate, so check overlap
    overlap = len(set(flat_result) & set(hnsw_result))
    assert overlap >= 3, f"Expected at least 3 matches, got {overlap}"


# ========== Edge Case Tests ==========


def test_flat_index_high_dimensional():
    """Test FaissFlatIndex with high-dimensional data."""
    np.random.seed(42)
    data = _make_data(n=100, d=128, seed=1)
    query = _make_query(d=128, seed=2)
    
    k = 5
    index = FaissFlatIndex().build(data, metric="l2")
    result = index.knn(query, k=k)
    
    assert len(result) == k + 10
    expected = _true_topk_l2(data, query, k=k)
    for neighbor in expected:
        assert neighbor in result


def test_flat_index_query_is_in_data():
    """Test FaissFlatIndex when query is identical to a data point."""
    np.random.seed(42)
    data = _make_data(n=50, d=8, seed=1)
    query = data[10].copy()  # Use an existing point as query
    
    index = FaissFlatIndex().build(data, metric="l2")
    result = index.knn(query, k=1)
    
    assert len(result) == 11  # k + 10
    assert result[0] == 10  # Should return the exact point first


def test_indices_are_unique():
    """Test that returned indices are unique (no duplicates)."""
    np.random.seed(42)
    data = _make_data(n=100, d=10, seed=1)
    query = _make_query(d=10, seed=2)
    
    k = 10
    index = FaissFlatIndex().build(data, metric="l2")
    result = index.knn(query, k=k)
    
    assert len(result) == k + 10
    assert len(result) == len(set(result)), "Returned indices should be unique"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
