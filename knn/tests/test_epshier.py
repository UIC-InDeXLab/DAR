import numpy as np
import pytest

try:
    # When running from repo root
    from knn.methods.epshier import (
        EpsHierIndex,
        compute_sample_size,
        random_sample,
        Config,
    )
except Exception:  # pragma: no cover
    # When running from within knn/ (so `methods` is importable)
    from methods.epshier import EpsHierIndex, compute_sample_size, random_sample, Config


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def _make_data(n: int = 256, d: int = 8, seed: int = 0) -> np.ndarray:
    # faiss.Kmeans expects float32.
    data = _rng(seed).normal(size=(n, d)).astype(np.float32)
    return data


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _make_query(d: int, seed: int) -> np.ndarray:
    q = _rng(seed).normal(size=(d,)).astype(np.float32)
    norm = np.linalg.norm(q)
    if norm == 0:
        q[0] = 1.0
        norm = 1.0
    return q / norm


def _true_topk_indices(data: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    scores = data @ query
    # descending by score
    return np.argsort(-scores)[:k]


def _true_kth_index(data: np.ndarray, query: np.ndarray, k: int) -> int:
    scores = data @ query
    # descending by score; k is 1-based
    return int(np.argsort(-scores)[k - 1])


def _brute_stripe_indices(data: np.ndarray, query: np.ndarray, low: float, high: float):
    scores = data @ query
    return np.nonzero((scores >= low) & (scores <= high))[0].astype(np.int64)


def test_random_sample_size_ge_n_returns_all_points():
    np.random.seed(0)
    pts = _make_data(n=32, d=3, seed=10)
    idx, sample = random_sample(pts, size=10_000)

    assert idx.dtype == np.int64 or np.issubdtype(idx.dtype, np.integer)
    assert idx.shape == (pts.shape[0],)
    assert np.array_equal(idx, np.arange(len(pts)))
    assert sample.shape == pts.shape
    assert np.all(sample == pts)


def test_random_sample_size_lt_n_is_subset_without_replacement():
    np.random.seed(123)
    pts = _make_data(n=50, d=2, seed=11)
    idx, sample = random_sample(pts, size=7)

    assert idx.shape == (7,)
    assert sample.shape == (7, 2)
    assert len(set(map(int, idx.tolist()))) == 7
    assert np.all((0 <= idx) & (idx < len(pts)))
    assert np.all(sample == pts[idx])


def test_compute_sample_size_monotonic_in_epsilon_and_dimension():
    # Smaller epsilon => larger sample.
    m1 = compute_sample_size(d=8, epsilon=0.5)
    m2 = compute_sample_size(d=8, epsilon=0.25)
    m3 = compute_sample_size(d=8, epsilon=0.125)
    assert m1 <= m2 <= m3

    # Higher dimension => larger sample (vc_dim = d+1).
    m4 = compute_sample_size(d=4, epsilon=0.25)
    m5 = compute_sample_size(d=16, epsilon=0.25)
    assert m4 <= m5


@pytest.mark.parametrize("n,d,num_levels,branching", [(128, 4, 6, 4), (256, 8, 5, 8)])
def test_build_process_creates_consistent_hierarchy(
    n: int, d: int, num_levels: int, branching: int
):
    np.random.seed(123)  # controls eps-sample selection in random_sample
    data = _make_data(n=n, d=d, seed=1)
    config = Config(num_levels=num_levels, branching_factor=branching, eps=0.01)
    # choose eps small enough so eps-sample size >= n (=> full sample), making the stripe deterministic.
    idx = EpsHierIndex(config=config, verbose=False)
    idx.build_index(data)

    assert hasattr(idx, "data")
    assert idx.data is data
    assert idx.data_size == n
    assert idx.dim == d

    # At least level 0 exists.
    assert len(idx.levels) >= 1

    level0 = idx.levels[0]
    assert level0.level_idx == 0
    assert level0.size == n
    assert level0.ball_centers.shape == (n, d)
    assert level0.ball_radii.shape == (n,)

    # For every non-root level i, child2parent should map its items to next level's items.
    for i in range(len(idx.levels) - 1):
        child = idx.levels[i]
        parent = idx.levels[i + 1]

        assert child.child2parent is not None
        assert child.num_parents == parent.size
        assert child.child2parent.shape == (child.size,)
        assert child.child2parent.min() >= 0
        assert child.child2parent.max() < parent.size

        assert child.parent_mask_buf is not None
        assert child.parent_mask_buf.shape == (child.num_parents,)
        assert child.parent_mask_buf.dtype == bool

        assert np.all(child.ball_radii >= 0)
        assert np.all(parent.ball_radii >= 0)

    # Root has no parent pointers.
    root = idx.levels[-1]
    assert root.child2parent is None


def test_next_level_unit_when_level_size_ge_num_points():
    pts = _make_data(n=10, d=3, seed=12)
    config = Config(num_levels=2, branching_factor=4, eps=0.1)
    idx = EpsHierIndex(config=config)
    idx.dim = pts.shape[1]
    centers, radii, assign = idx._next_level(pts, level_size=10)

    assert centers.shape == pts.shape
    assert np.all(centers == pts)
    assert radii.shape == (10,)
    assert np.all(radii == 0)
    assert assign.shape == (10,)
    assert np.array_equal(assign, np.arange(10))


def test_refined_radii_upper_bound_children():
    np.random.seed(123)
    data = _make_data(n=192, d=6, seed=2)
    config = Config(num_levels=6, branching_factor=4, eps=0.01)
    idx = EpsHierIndex(config=config, verbose=False).build_index(data)

    # For each parent ball, ensure all its child balls are contained:
    # ||c_child - c_parent|| + r_child <= r_parent
    for i in range(len(idx.levels) - 1):
        child = idx.levels[i]
        parent = idx.levels[i + 1]

        assert child.child2parent is not None
        assignments = child.child2parent

        for parent_idx in range(parent.size):
            children = np.where(assignments == parent_idx)[0]
            if children.size == 0:
                continue

            child_centers = child.ball_centers[children]
            child_radii = child.ball_radii[children]
            dists = np.linalg.norm(
                child_centers - parent.ball_centers[parent_idx], axis=1
            )

            assert np.all(dists + child_radii <= parent.ball_radii[parent_idx] + 1e-5)


@pytest.mark.parametrize("k", [1, 3, 10, 25])
def test_query_has_100_percent_recall_of_true_kth(k: int):
    np.random.seed(123)
    data = _make_data(n=256, d=8, seed=3)

    # eps small => eps-sample becomes full data, so stripe should bracket kth rank reliably.
    config = Config(num_levels=6, branching_factor=4, eps=0.01)
    idx = EpsHierIndex(config=config, verbose=False).build_index(data)

    # multiple queries to exercise different traversal paths
    for qseed in range(10):
        q = _make_query(d=data.shape[1], seed=100 + qseed)

        true_kth = _true_kth_index(data, q, k)
        candidates = idx.query(q, k)

        assert candidates.dtype == np.int64
        assert candidates.ndim == 1
        assert np.all((0 <= candidates) & (candidates < data.shape[0]))

        cand_set = set(map(int, candidates.tolist()))
        assert int(true_kth) in cand_set


def test_query_handles_k_extremes_and_is_repeatable():
    np.random.seed(123)
    data = _make_data(n=128, d=6, seed=13)
    config = Config(num_levels=6, branching_factor=4, eps=0.01)
    idx = EpsHierIndex(config=config, verbose=False).build_index(data)

    q = _make_query(d=data.shape[1], seed=2026)

    for k in [1, data.shape[0]]:
        true_kth = _true_kth_index(data, q, k)

        out1 = idx.query(q, k)
        out2 = idx.query(q, k)

        assert np.array_equal(out1, out2)
        assert int(true_kth) in set(map(int, out1.tolist()))


def test_query_candidates_are_subset_of_stripe_exact_check():
    np.random.seed(123)
    data = _make_data(n=200, d=5, seed=4)
    config = Config(num_levels=6, branching_factor=4, eps=0.01)
    idx = EpsHierIndex(config=config, verbose=False).build_index(data)

    q = _make_query(d=data.shape[1], seed=999)
    k = 17

    _, low, high = idx._find_stripe(q, k)
    scores = data @ q
    brute_stripe = set(np.nonzero((scores >= low) & (scores <= high))[0].tolist())

    candidates = set(map(int, idx.query(q, k).tolist()))

    # The hierarchical pruning should be conservative (no false negatives w.r.t. stripe).
    assert candidates.issubset(brute_stripe)


def test_find_stripe_bounds_ordered_and_in_range():
    np.random.seed(123)
    data = _make_data(n=128, d=4, seed=14)
    config = Config(num_levels=5, branching_factor=4, eps=0.01)
    idx = EpsHierIndex(config=config, verbose=False).build_index(data)

    q = _make_query(d=data.shape[1], seed=15)
    scores = data @ q
    mn, mx = float(scores.min()), float(scores.max())

    for k in [1, data.shape[0] // 2, data.shape[0]]:
        _, low, high = idx._find_stripe(q, k)
        assert low <= high
        assert mn - 1e-6 <= low <= mx + 1e-6
        assert mn - 1e-6 <= high <= mx + 1e-6


def test_exact_filter_chunked_matches_bruteforce():
    np.random.seed(123)
    data = _make_data(n=180, d=7, seed=16)
    config = Config(num_levels=6, branching_factor=4, eps=0.01)
    idx = EpsHierIndex(config=config, verbose=False).build_index(data)

    q = _make_query(d=data.shape[1], seed=17)
    k = 33
    _, low, high = idx._find_stripe(q, k)

    # pick an arbitrary subset to simulate leaf candidates
    leaf_idx = np.array(sorted(_rng(18).choice(data.shape[0], size=57, replace=False)))

    brute = set(int(x) for x in leaf_idx if low <= float((data[int(x)] @ q)) <= high)

    filtered = idx.exact_filter_chunked(data, leaf_idx, q, low, high, chunk=8)
    assert set(map(int, filtered.tolist())) == brute


@pytest.mark.parametrize("k", [1, 5, 20, 50])
def test_query_cosine_similarity_contains_true_kth_when_data_unit_normalized(k: int):
    """Cosine similarity is dot-product when both vectors are unit-normalized."""
    np.random.seed(123)
    raw = _make_data(n=256, d=8, seed=101)
    data = _normalize_rows(raw)

    # eps small => eps-sample becomes full data, so stripe should bracket kth rank reliably.
    config = Config(num_levels=6, branching_factor=4, eps=0.01)
    idx = EpsHierIndex(config=config, verbose=False).build_index(data)

    for qseed in range(10):
        q = _make_query(d=data.shape[1], seed=200 + qseed)  # already unit-normalized

        # brute-force cosine == dot for unit vectors
        scores = data @ q
        true_kth = int(np.argsort(-scores)[k - 1])

        candidates = idx.query(q, k)
        assert candidates.dtype == np.int64
        assert candidates.ndim == 1
        assert np.all((0 <= candidates) & (candidates < data.shape[0]))
        assert true_kth in set(map(int, candidates.tolist()))


def test_find_stripe_cosine_bounds_within_minus1_plus1_for_unit_vectors():
    np.random.seed(123)
    raw = _make_data(n=128, d=6, seed=303)
    data = _normalize_rows(raw)

    config = Config(num_levels=5, branching_factor=4, eps=0.01)
    idx = EpsHierIndex(config=config, verbose=False).build_index(data)

    q = _make_query(d=data.shape[1], seed=304)  # unit-normalized

    # cosine similarity is in [-1, 1] for unit vectors
    for k in [1, data.shape[0] // 2, data.shape[0]]:
        _, low, high = idx._find_stripe(q, k)
        assert -1.0 - 1e-6 <= low <= 1.0 + 1e-6
        assert -1.0 - 1e-6 <= high <= 1.0 + 1e-6
        assert low <= high
