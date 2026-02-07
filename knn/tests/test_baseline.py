import numpy as np
import pytest

try:
    # When running from repo root
    from knn.methods.baseline import calc_score, quickselect
except Exception:  # pragma: no cover
    # When running from within knn/
    from methods.baseline import calc_score, quickselect


def _expected_kth_index(points: np.ndarray, q: np.ndarray, k: int, metric: str) -> int:
    scores = np.array([calc_score(p, q, metric) for p in points], dtype=float)
    idx = np.arange(points.shape[0], dtype=int)
    order = np.lexsort((idx, scores))  # sort by score, then by index
    return int(order[k])


def _expected_kth_score(points: np.ndarray, q: np.ndarray, k: int, metric: str) -> float:
    scores = np.array([calc_score(p, q, metric) for p in points], dtype=float)
    return float(np.sort(scores)[k])


def test_quickselect_l2_unique_distances():
    pts = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    q = np.array([0.0], dtype=np.float32)

    for k in range(len(pts)):
        got = quickselect(pts, k, q, metric="l2")
        exp = _expected_kth_index(pts, q, k, metric="l2")
        assert got == exp


def test_quickselect_l2_ties_returns_smallest_index():
    # Distances to q: [0, 1, 1, 2] => tie at score=1 between indices 1 and 2.
    pts = np.array([[0.0], [1.0], [1.0], [2.0]], dtype=np.float32)
    q = np.array([0.0], dtype=np.float32)

    got = quickselect(pts, 1, q, metric="l2")
    exp = _expected_kth_index(pts, q, 1, metric="l2")
    assert got == exp
    assert got == 1


def test_quickselect_cosine_basic_ordering():
    # Query points along +x.
    pts = np.array(
        [
            [1.0, 0.0],  # cos=1
            [0.0, 1.0],  # cos=0
            [-1.0, 0.0],  # cos=-1
        ],
        dtype=np.float32,
    )
    q = np.array([1.0, 0.0], dtype=np.float32)

    # calc_score returns cosine distance = 1 - cosine_similarity (smaller is closer).
    # cos values: [1, 0, -1] => dist values: [0 (idx0), 1 (idx1), 2 (idx2)]
    assert quickselect(pts, 0, q, metric="cosine") == 0
    assert quickselect(pts, 1, q, metric="cosine") == 1
    assert quickselect(pts, 2, q, metric="cosine") == 2


def test_quickselect_cosine_ties_returns_smallest_index():
    # Indices 0 and 1 both have cos=1 with q (same direction), so tie on score=-1.
    pts = np.array(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    q = np.array([1.0, 0.0], dtype=np.float32)

    # cos values: [1, 1, 0] => dist values: [0, 0, 1]
    # For k=1, the kth score is 0 (tied). quickselect returns the smallest index among ties.
    got = quickselect(pts, 1, q, metric="cosine")
    exp_score = _expected_kth_score(pts, q, 1, metric="cosine")
    got_score = float(calc_score(pts[got], q, metric="cosine"))
    assert got_score == exp_score
    assert got == 0


def test_calc_score_cosine_zero_norm_is_inf():
    pt = np.array([0.0, 0.0], dtype=np.float32)
    q = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ZeroDivisionError):
        calc_score(pt, q, metric="cosine")

    pt2 = np.array([1.0, 0.0], dtype=np.float32)
    q2 = np.array([0.0, 0.0], dtype=np.float32)
    with pytest.raises(ZeroDivisionError):
        calc_score(pt2, q2, metric="cosine")


def test_quickselect_cosine_zero_norm_raises():
    pts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    q = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ZeroDivisionError):
        quickselect(pts, 0, q, metric="cosine")


@pytest.mark.parametrize("k", [-1, 4])
def test_quickselect_rejects_out_of_bounds_k(k: int):
    pts = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    q = np.array([0.0], dtype=np.float32)
    with pytest.raises(ValueError):
        quickselect(pts, k, q, metric="l2")


def test_quickselect_rejects_unknown_metric():
    pts = np.array([[0.0], [1.0]], dtype=np.float32)
    q = np.array([0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Unknown metric"):
        quickselect(pts, 0, q, metric="nope")


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_quickselect_randomized_l2_matches_exact_kth_score(seed: int):
    rng = np.random.default_rng(seed)
    n, d = 256, 16
    pts = rng.normal(size=(n, d)).astype(np.float32)
    # Inject duplicates to force ties.
    pts[10] = pts[0]
    pts[11] = pts[0]
    pts[12] = pts[1]
    q = rng.normal(size=(d,)).astype(np.float32)

    for k in [0, 1, 2, 5, 10, 50, 100, n - 1]:
        idx = quickselect(pts, k, q, metric="l2")
        got = float(calc_score(pts[idx], q, "l2"))
        exp = _expected_kth_score(pts, q, k, metric="l2")
        assert got == exp


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_quickselect_randomized_cosine_matches_exact_kth_score(seed: int):
    rng = np.random.default_rng(seed)
    n, d = 256, 32
    pts = rng.normal(size=(n, d)).astype(np.float32)
    # Force duplicates for ties.
    pts[10] = pts[5]
    pts[11] = pts[5]

    # Ensure non-zero norms to avoid ZeroDivisionError in calc_score.
    q = rng.normal(size=(d,)).astype(np.float32)
    if not np.any(q):
        q[0] = 1.0

    # Replace any accidental zero-norm rows (extremely unlikely, but keep the test robust).
    norms = np.linalg.norm(pts, axis=1)
    zero_rows = np.where(norms == 0)[0]
    for i in zero_rows:
        pts[i, 0] = 1.0

    for k in [0, 1, 2, 5, 10, 50, 100, n - 1]:
        idx = quickselect(pts, k, q, metric="cosine")
        got = float(calc_score(pts[idx], q, "cosine"))
        exp = _expected_kth_score(pts, q, k, metric="cosine")
        assert got == exp


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_quickselect_tie_breaks_to_smallest_index(seed: int):
    rng = np.random.default_rng(seed)
    n, d = 128, 8
    pts = rng.normal(size=(n, d)).astype(np.float32)
    q = rng.normal(size=(d,)).astype(np.float32)

    # Create a tie at an arbitrary score by duplicating a point.
    pts[20] = pts[7]
    pts[21] = pts[7]

    scores = np.array([calc_score(p, q, "l2") for p in pts], dtype=float)
    order = np.lexsort((np.arange(n), scores))
    sorted_scores = scores[order]

    tie_ks = np.where(sorted_scores[:-1] == sorted_scores[1:])[0]
    assert tie_ks.size > 0

    # pick the second element in the first tied pair
    k = int(tie_ks[0] + 1)

    idx = quickselect(pts, k, q, metric="l2")
    kth_score = float(sorted_scores[k])
    tied = np.where(scores == kth_score)[0]
    assert idx == int(tied.min())
