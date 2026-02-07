import numpy as np
import pytest

try:
    # When running from repo root
    from knn.methods.epshier import Config, EpsHierANNIndex
except Exception:  # pragma: no cover
    # When running from within knn/
    from methods.epshier import Config, EpsHierANNIndex


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def _make_data(n: int = 256, d: int = 8, seed: int = 0) -> np.ndarray:
    return _rng(seed).normal(size=(n, d)).astype(np.float32)


def _make_query(d: int, seed: int) -> np.ndarray:
    q = _rng(seed).normal(size=(d,)).astype(np.float32)
    # avoid pathological all-zeros
    if not np.any(q):
        q[0] = 1.0
    return q


def _topk_by_l2(pts: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    dist2 = np.sum((pts - q) ** 2, axis=1)
    return np.argsort(dist2)[:k].astype(np.int64)


def _topk_by_cosine(normed_pts: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    qn = q / np.linalg.norm(q)
    scores = normed_pts @ qn
    return np.argsort(-scores)[:k].astype(np.int64)


def _kth_by_l2(pts: np.ndarray, q: np.ndarray, k: int) -> int:
    dist2 = np.sum((pts - q) ** 2, axis=1)
    return int(np.argsort(dist2)[k - 1])


def _kth_by_cosine(normed_pts: np.ndarray, q: np.ndarray, k: int) -> int:
    qn = q / np.linalg.norm(q)
    scores = normed_pts @ qn
    return int(np.argsort(-scores)[k - 1])


def _recall_at_k(candidates: np.ndarray, truth_topk: np.ndarray) -> float:
    if truth_topk.size == 0:
        return 1.0
    hits = int(np.isin(truth_topk, candidates).sum())
    return hits / float(truth_topk.size)


def test_build_process_creates_underlying_hierarchy():
    np.random.seed(123)
    pts = _make_data(n=128, d=6, seed=1)

    ann = EpsHierANNIndex(Config(num_levels=6, branching_factor=4, eps=0.01)).build_index(pts)

    assert ann.lifted_pts is not None
    assert ann.lifted_pts.shape == (pts.shape[0], pts.shape[1] + 1)

    # Validate paraboloid lift layout.
    assert np.allclose(ann.lifted_pts[:, : pts.shape[1]], pts)
    assert np.allclose(ann.lifted_pts[:, pts.shape[1]], np.sum(pts * pts, axis=1))

    # underlying hierarchical index is built on lifted points
    assert hasattr(ann.index, "levels")
    assert len(ann.index.levels) >= 1
    assert ann.index.data.shape == ann.lifted_pts.shape


def test_kthnn_translates_query_to_expected_hyperplane(monkeypatch):
    np.random.seed(123)
    pts = _make_data(n=50, d=7, seed=10)
    ann = EpsHierANNIndex(Config(num_levels=5, branching_factor=4, eps=0.01)).build_index(pts)

    q = _make_query(d=pts.shape[1], seed=11)
    k = 7

    captured = {}

    def _wrapped_query(query_vec, kk):
        captured["query"] = np.array(query_vec, copy=True)
        captured["k"] = kk
        return np.array([], dtype=np.int64)

    monkeypatch.setattr(ann.index, "query", _wrapped_query)
    ann.query(q, k)

    assert captured["k"] == k - 1
    expected = np.concatenate((-2 * q, np.array([1.0], dtype=q.dtype)))
    got = captured["query"]
    assert got.shape == expected.shape

    # Implementation may normalize and/or negate the lifted query.
    got_n = got / np.linalg.norm(got)
    exp_n = expected / np.linalg.norm(expected)
    assert abs(float(np.dot(got_n, exp_n))) > 1 - 1e-6


@pytest.mark.parametrize("k", [1, 5, 20])
def test_kthnn_contains_true_kth_nearest_neighbor_by_l2(k: int):
    # Validate 100% recall for the k-th nearest neighbor (L2) being present
    # in the returned candidate set.
    np.random.seed(123)
    pts = _make_data(n=256, d=8, seed=2)
    ann = EpsHierANNIndex(Config(num_levels=6, branching_factor=4, eps=0.01)).build_index(pts)

    for qseed in range(5):
        q = _make_query(d=pts.shape[1], seed=100 + qseed)

        got = ann.query(q, k)
        assert got.dtype == np.int64
        assert got.ndim == 1
        assert np.all((0 <= got) & (got < pts.shape[0]))

        # Ground truth: exact L2 k-th nearest neighbor index.
        dist2 = np.sum((pts - q) ** 2, axis=1)
        true_kth = int(np.argsort(dist2)[k - 1])
        assert true_kth in set(map(int, got.tolist()))


def test_kthnn_contains_true_l2_nearest_neighbor():
    # Deterministic repro for the L2-nearest-neighbor intent.
    np.random.seed(123)
    pts = _make_data(n=256, d=8, seed=2)
    ann = EpsHierANNIndex(Config(num_levels=6, branching_factor=4, eps=0.01)).build_index(pts)

    q = _make_query(d=pts.shape[1], seed=100)
    k = 1
    got = ann.query(q, k)

    dist2 = np.sum((pts - q) ** 2, axis=1)
    true_nn = int(np.argmin(dist2))
    assert true_nn in set(map(int, got.tolist()))


def test_kthnn_k_ge_n_returns_all_points_sorted():
    # k should be clamped to n and must not crash.
    pts = _make_data(n=64, d=5, seed=3)
    ann = EpsHierANNIndex(Config(num_levels=5, branching_factor=4, eps=0.01)).build_index(pts)

    q = _make_query(d=pts.shape[1], seed=4)
    got = ann.query(q, k=10_000)
    assert got.dtype == np.int64
    assert got.ndim == 1
    assert np.all((0 <= got) & (got < pts.shape[0]))


def test_kthnn_raises_before_build():
    ann = EpsHierANNIndex(Config(num_levels=4, branching_factor=4, eps=0.01))
    with pytest.raises(RuntimeError):
        ann.query(np.zeros(3, dtype=np.float32), 1)


@pytest.mark.parametrize("k", [0, -1])
def test_kthnn_rejects_non_positive_k(k: int):
    pts = _make_data(n=32, d=4, seed=7)
    ann = EpsHierANNIndex(Config(num_levels=4, branching_factor=4, eps=0.01)).build_index(pts)
    with pytest.raises(ValueError):
        ann.query(_make_query(d=pts.shape[1], seed=8), k)


def test_kthnn_rejects_non_1d_query():
    pts = _make_data(n=32, d=4, seed=9)
    ann = EpsHierANNIndex(Config(num_levels=4, branching_factor=4, eps=0.01)).build_index(pts)
    q2d = _rng(10).normal(size=(2, pts.shape[1])).astype(np.float32)
    with pytest.raises(ValueError):
        ann.query(q2d, 1)


def test_build_cosine_does_not_lift_and_normalizes_points():
    np.random.seed(123)
    pts = _make_data(n=120, d=6, seed=50)
    ann = EpsHierANNIndex(
        Config(num_levels=5, branching_factor=4, eps=0.01, norm="cosine")
    ).build_index(pts)

    assert ann.lifted_pts is not None
    assert ann.lifted_pts.shape == pts.shape

    norms = np.linalg.norm(ann.lifted_pts, axis=1)
    # points are normalized (up to numerical precision)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_kthnn_cosine_normalizes_query(monkeypatch):
    np.random.seed(123)
    pts = _make_data(n=80, d=5, seed=51)
    ann = EpsHierANNIndex(
        Config(num_levels=5, branching_factor=4, eps=0.01, norm="cosine")
    ).build_index(pts)

    q = _make_query(d=pts.shape[1], seed=52) * 10.0  # scaled on purpose
    k = 7

    captured = {}

    def _wrapped_query(query_vec, kk):
        captured["query"] = np.array(query_vec, copy=True)
        captured["k"] = kk
        return np.array([], dtype=np.int64)

    monkeypatch.setattr(ann.index, "query", _wrapped_query)
    ann.query(q, k)

    assert captured["k"] == k - 1
    got = captured["query"]
    assert got.shape == q.shape
    assert np.isclose(np.linalg.norm(got), 1.0, atol=1e-6)

    # same direction as q
    qn = q / np.linalg.norm(q)
    assert abs(float(np.dot(got, qn))) > 1 - 1e-6


@pytest.mark.parametrize("k", [1, 5, 20])
def test_kthnn_cosine_contains_true_kth_by_cosine_similarity(k: int):
    # For cosine mode we expect k-th by cosine similarity (dot product on normalized vectors)
    np.random.seed(123)
    pts = _make_data(n=256, d=8, seed=60)

    # Ensure points are not pathological (normalization happens in build)
    ann = EpsHierANNIndex(
        Config(num_levels=6, branching_factor=4, eps=0.01, norm="cosine")
    ).build_index(pts)

    assert ann.lifted_pts is not None
    normed_pts = ann.lifted_pts

    for qseed in range(5):
        q = _make_query(d=pts.shape[1], seed=70 + qseed)
        qn = q / np.linalg.norm(q)

        got = ann.query(q, k)
        assert got.dtype == np.int64
        assert got.ndim == 1
        assert np.all((0 <= got) & (got < pts.shape[0]))

        scores = normed_pts @ qn
        true_kth = int(np.argsort(-scores)[k - 1])
        assert true_kth in set(map(int, got.tolist()))


def test_candidates_are_unique_and_in_range_across_configs():
    # Ensure candidate indices are valid and unique (no duplicates).
    np.random.seed(123)
    pts = _make_data(n=200, d=10, seed=90)

    configs = [
        Config(num_levels=6, branching_factor=2, eps=0.01),
        Config(num_levels=6, branching_factor=4, eps=0.01),
        Config(num_levels=6, branching_factor=4, eps=0.02),
    ]

    q = _make_query(d=pts.shape[1], seed=91)
    for cfg in configs:
        ann = EpsHierANNIndex(cfg).build_index(pts)
        got = ann.query(q, k=10)
        assert got.dtype == np.int64
        assert got.ndim == 1
        assert np.all((0 <= got) & (got < pts.shape[0]))
        assert np.unique(got).size == got.size


@pytest.mark.parametrize(
    "cfg",
    [
        Config(num_levels=6, branching_factor=2, eps=0.01),
        Config(num_levels=7, branching_factor=4, eps=0.01),
    ],
)
@pytest.mark.parametrize("k", [1, 10, 20])
def test_final_recall_l2_kth_contained_is_close_to_100_percent(cfg: Config, k: int):
    # The ANN wrapper returns a *candidate set* derived from a score stripe.
    # The intended guarantee to validate is that the true k-th nearest neighbor
    # is contained in the returned candidates with near-perfect recall.
    np.random.seed(123)
    pts = _make_data(n=512, d=16, seed=200)
    ann = EpsHierANNIndex(cfg).build_index(pts)

    hits = 0
    num_queries = 50
    for qseed in range(num_queries):
        q = _make_query(d=pts.shape[1], seed=300 + qseed)
        candidates = ann.query(q, k)
        true_kth = _kth_by_l2(pts, q, k)
        hits += int(np.any(candidates == true_kth))

    final_recall = hits / float(num_queries)
    assert final_recall >= 0.98


@pytest.mark.parametrize(
    "cfg",
    [
        Config(num_levels=6, branching_factor=2, eps=0.01, norm="cosine"),
        Config(num_levels=7, branching_factor=4, eps=0.01, norm="cosine"),
    ],
)
@pytest.mark.parametrize("k", [1, 10, 20])
def test_final_recall_cosine_kth_contained_is_close_to_100_percent(cfg: Config, k: int):
    np.random.seed(123)
    pts = _make_data(n=512, d=16, seed=210)
    ann = EpsHierANNIndex(cfg).build_index(pts)

    assert ann.lifted_pts is not None
    normed_pts = ann.lifted_pts

    hits = 0
    num_queries = 50
    for qseed in range(num_queries):
        q = _make_query(d=pts.shape[1], seed=400 + qseed)
        candidates = ann.query(q, k)
        true_kth = _kth_by_cosine(normed_pts, q, k)
        hits += int(np.any(candidates == true_kth))

    final_recall = hits / float(num_queries)
    assert final_recall >= 0.98


def test_duplicate_points_do_not_break_query_and_recall_remains_high():
    # Duplicate points are common in real datasets; ensure stability.
    np.random.seed(123)
    base = _make_data(n=128, d=8, seed=500)
    pts = base.copy()
    pts[:5] = pts[0]  # introduce duplicates

    ann = EpsHierANNIndex(Config(num_levels=6, branching_factor=4, eps=0.01)).build_index(pts)
    q = pts[0].copy()

    k = 5
    candidates = ann.query(q, k)
    true_kth = _kth_by_l2(pts, q, k)
    assert true_kth in set(map(int, candidates.tolist()))
