import numpy as np
from dataclasses import dataclass
from typing import Optional
import faiss
import math
from tqdm import tqdm

try:  # optional (numba)
    from .numba_kernels import exact_filter_range_mask_numba
except Exception:  # pragma: no cover
    exact_filter_range_mask_numba = None


def random_sample(points: np.ndarray, size: int):
    # random sample from points
    n = len(points)
    if size >= n:
        return np.arange(n), points
    indices = np.random.choice(n, size=size, replace=False)
    return indices, points[indices]


def compute_sample_size(d, epsilon, delta=0.1, constant=1 / 2**7):
    vc_dim = d + 1
    m = (constant / (epsilon**2)) * (
        vc_dim * math.log(1 / epsilon) + math.log(1 / delta)
    )
    return math.ceil(m)


@dataclass
class EpsHierConfig:
    num_levels: int
    branching_factor: int
    eps: float
    norm: str = "l2"  # cosine
    eps_sample_constant: float = 1 / 2**7
    kmeans_niter: int = 1


# Backwards-compatible alias used by tests/clients.
Config = EpsHierConfig


class EpsHierIndex:

    @dataclass
    class Level:
        level_idx: int
        ball_centers: np.ndarray  # center of balls
        ball_radii: np.ndarray  # radius of balls
        size: int

        # store parent-child relationships (THIS IS CHILD LEVEL)
        # local child idx (current level) -> local parent idx
        child2parent: Optional[np.ndarray]
        num_parents: Optional[int] = None
        parent_mask_buf: Optional[np.ndarray] = None

    def __init__(self, config: "EpsHierConfig", verbose=False):
        self.levels = []
        self.eps_sample_idx = None
        self.eps_sample = None
        self.num_levels = config.num_levels
        self.branching_factor = config.branching_factor
        self.verbose = verbose
        self.eps = config.eps
        self.norm = config.norm
        self.constant = config.eps_sample_constant
        self.kmeans_niter = int(getattr(config, "kmeans_niter", 1))

    def build_index(self, data: np.ndarray):
        # find an eps sample
        _, self.eps_sample = random_sample(
            data, compute_sample_size(data.shape[1], self.eps, constant=self.constant)
        )

        if self.verbose:
            print(f"Eps sample size: {len(self.eps_sample)} / {len(data)}")

        # build hierarchical index on data
        self.data = data
        self.data_size, self.dim = data.shape
        self.eps_sample_size = len(self.eps_sample)

        level_size = self.data_size
        level_idx = 0
        ball_centers = self.data

        if self.verbose:
            print(f"Building level {level_idx} with size {level_size}")

        level_0 = self.Level(
            level_idx=level_idx,
            ball_centers=ball_centers,
            ball_radii=np.zeros(level_size),
            size=level_size,
            child2parent=None,
            num_parents=None,
        )
        self.levels.append(level_0)

        while True:
            level_size = level_size // self.branching_factor

            if self.verbose:
                print(f"Building level {level_idx} with size {level_size}")

            # check the finish
            if level_size <= 1 or level_idx + 1 >= self.num_levels:
                break

            (ball_centers, ball_radii, assignments) = self._next_level(
                ball_centers, level_size  # type: ignore
            )

            child_level = self.levels[level_idx]
            child_level.num_parents = level_size

            child_level.child2parent = assignments

            ball_radii = self._refine_ball_radii(child_level, assignments, ball_centers)

            # Create new level
            parent_level = EpsHierIndex.Level(
                level_idx=level_idx + 1,
                ball_centers=ball_centers,  # type: ignore
                ball_radii=ball_radii,
                size=len(ball_centers),  # type: ignore
                # child <-> parent
                child2parent=None,  # to be filled later
            )
            self.levels.append(parent_level)

            level_idx += 1

        for lvl in self.levels:
            if lvl.num_parents is not None:
                lvl.parent_mask_buf = np.zeros(lvl.num_parents, dtype=bool)

        return self

    def _refine_ball_radii(self, child_level, assignments, ball_centers):
        ball_radii = np.zeros(len(ball_centers))

        for parent_idx, ball_center in enumerate(ball_centers):  # type: ignore
            # children of this parent
            children = np.where(assignments == parent_idx)[0]

            if len(children) == 0:
                ball_radii[parent_idx] = 0.0
                continue

            child_centers = child_level.ball_centers[children]
            child_radii = child_level.ball_radii[children]

            # get max of ||child_center - parent_center|| + child_radius
            dists = np.linalg.norm(child_centers - ball_center, axis=1)
            refined_radius = np.max(dists + child_radii)
            ball_radii[parent_idx] = refined_radius

        return ball_radii

    def _next_level(self, points: np.ndarray, level_size: int):
        # verify num_centroids
        if level_size >= points.shape[0]:
            # each point is its own cluster
            p2cluster = np.array([i for i in range(len(points))])
            ball_centers = points
            ball_radii = np.zeros(len(points))
            return ball_centers, ball_radii, p2cluster

        kmeans = faiss.Kmeans(
            d=self.dim,
            k=level_size,
            niter=self.kmeans_niter,
            verbose=self.verbose,
            gpu=True,
            spherical=self.norm == "cosine",
        )
        if self.verbose:
            print(f"Training kmeans... {self.norm}")
        kmeans.train(points)
        _, assignments = kmeans.index.search(points, 1)  # type: ignore

        centroids = kmeans.centroids
        # assignment[i] = j means points[i] assigned to cluster j
        assignments = assignments.squeeze()

        ball_radii = np.zeros(level_size)

        for cluster_idx, centroid in enumerate(centroids):  # type: ignore
            cluster_points_indexes = (assignments == cluster_idx).nonzero()[0]

            if cluster_points_indexes.size == 0:
                ball_radii[cluster_idx] = 0.0
                continue

            cluster_members = points[cluster_points_indexes]
            dists = np.linalg.norm(cluster_members - centroid, axis=1)
            ball_radii[cluster_idx] = dists.max()

        return centroids, ball_radii, assignments

    def query(self, query: np.ndarray, k: int):
        # The pruning logic treats each cluster as an L2 ball around its center.
        # For any fixed query vector q, we have (Cauchy-Schwarz):
        #   |q·x - q·c| = |q·(x-c)| <= ||q|| * ||x-c||
        # So an L2 radius r becomes an inner-product interval of +/- (||q|| * r).
        _, low_dot, high_dot = self._find_stripe(query, k)

        # search top-down
        # search root
        root = self.levels[-1]

        scores = np.dot(root.ball_centers, query)
        # between low and high
        # root_r = root.ball_radii * q_norm
        mask = (scores + root.ball_radii) >= low_dot
        mask &= (scores - root.ball_radii) <= high_dot
        active_cluster_idx = np.nonzero(mask)[0]

        # recursively search down the tree
        levels = self.levels[::-1][1:-1]  # skip root, reverse order, skip level 0
        for level in levels:
            # if self.verbose:
            #     print(
            #         f"Level {level.level_idx + 1} -> {level.level_idx}: {len(active_cluster_idx)} active clusters"
            #     )
            # level here is the child level of previous one

            if active_cluster_idx.size == 0:
                return np.array([], dtype=np.int64)

            p = level.child2parent  # [C]
            buf = level.parent_mask_buf
            if p is None or buf is None:
                raise RuntimeError(
                    "Index is not fully built (missing child2parent/mask buffer)."
                )
            buf.fill(False)
            buf[active_cluster_idx] = True
            child_idx = np.nonzero(buf[p])[0]

            child_radii = level.ball_radii[child_idx]
            child_centers = level.ball_centers[child_idx]

            # scores
            scores = np.dot(child_centers, query)
            # child_r = child_radii * q_norm
            mask = (scores + child_radii) >= low_dot
            mask &= (scores - child_radii) <= high_dot
            active_cluster_idx = child_idx[mask]

        # LEVEL 1 -> 0 (child2parent expansion)
        level0 = self.levels[0]  # keys
        buf = level0.parent_mask_buf
        if level0.child2parent is None or buf is None:
            raise RuntimeError(
                "Index is not fully built (missing level0 child2parent/mask buffer)."
            )
        buf.fill(False)
        buf[active_cluster_idx] = True
        leaf_idx = np.nonzero(buf[level0.child2parent])[0]

        # exact check
        qualifying_idx = self.exact_filter_chunked(
            self.data, leaf_idx, query, low_dot, high_dot
        )

        return qualifying_idx  # indices of keys satisfying q.k >= threshold

    def exact_filter_chunked(
        self,
        data: np.ndarray,
        leaf_idx: np.ndarray,
        query: np.ndarray,
        low_dot: float,
        high_dot: float,
        chunk: int = 1024 * 32,
        use_numba: bool = True,
    ):
        if leaf_idx.size == 0:
            return np.array([], dtype=np.int64)

        # If pruning is ineffective and leaf_idx contains (almost) the full dataset,
        # fancy indexing data[leaf_idx] will create a huge copy and be slow.
        # In that regime, compute scores for all points contiguously, then intersect
        # with leaf_idx to preserve EXACT baseline semantics.
        n_total = int(data.shape[0])
        if n_total > 0 and int(leaf_idx.size) >= int(0.95 * n_total):
            scores = data @ query
            keep = (scores >= low_dot) & (scores <= high_dot)
            in_leaf = np.zeros(n_total, dtype=bool)
            in_leaf[leaf_idx] = True
            return np.nonzero(keep & in_leaf)[0].astype(np.int64, copy=False)

        out = []
        for s in range(0, int(leaf_idx.size), int(chunk)):
            sub = leaf_idx[s : s + chunk]
            scores = data[sub] @ query
            keep = (scores >= low_dot) & (scores <= high_dot)
            if np.any(keep):
                out.append(sub[keep])
        return np.concatenate(out) if out else np.array([], dtype=np.int64)

    def _find_stripe(self, query, k):
        n = self.data_size
        m = len(self.eps_sample)

        scores = np.dot(self.eps_sample, query)

        rank_low = int(max(0, math.floor((k / n - self.eps) * m)))
        rank_high = int(min(m - 1, math.ceil((k / n + self.eps) * m)))

        # if self.verbose:
        #     print(f"k: {k}, n: {n}, m: {m}, eps: {self.eps}")
        #     print(f"Rank low: {rank_low}, Rank high: {rank_high} (m={m})")

        # Avoid full argsort: we only need two order statistics.
        # Use a single partition call to get both ranks.
        neg_scores = -scores
        neg_part = np.partition(neg_scores, (rank_low, rank_high))
        high = float(-neg_part[rank_low])
        low = float(-neg_part[rank_high])

        # Near the extremes (very small k or very large k), the epsilon-sample cannot
        # reliably bound the true maximum/minimum score because the extremal point is
        # unlikely to be present in the sample. In those cases, use a one-sided stripe.
        #
        # - If rank_low is clamped to 0 (k/n <= eps), drop the upper bound.
        # - If rank_high is clamped to m-1 (k/n >= 1-eps), drop the lower bound.
        if rank_low == 0:
            high = float("inf")
        if rank_high == m - 1:
            low = float("-inf")

        # if self.verbose:
        #     print(f"Initial stripe: low={low}, high={high}")
        #     scores = self.data @ query
        #     # filter by low and high
        #     exact_indices = np.nonzero((scores >= low) & (scores <= high))[0]
        #     print(f"Exact number of points in stripe: {len(exact_indices)}")

        return (query, low, high)


def paraboloid_lift(points: np.ndarray):
    n, d = points.shape
    lifted = np.zeros((n, d + 1), dtype=points.dtype)
    norms = np.linalg.norm(points, axis=1)
    lifted[:, :d] = points
    lifted[:, d] = norms**2
    return lifted


class EpsHierANNIndex:
    def __init__(self, config: "EpsHierConfig", verbose=False):
        self.index = EpsHierIndex(config, verbose=verbose)
        self.lifted_pts: Optional[np.ndarray] = None
        self.norm = config.norm

        if self.norm not in {"l2", "cosine"}:
            raise ValueError(f"Unsupported norm: {self.norm}. Use 'l2' or 'cosine'.")

    def build_index(self, pts: np.ndarray) -> "EpsHierANNIndex":
        if self.norm == "l2":
            self.lifted_pts = paraboloid_lift(pts)
        else:
            # cosine: normalize points so dot-product equals cosine similarity
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            self.lifted_pts = pts / norms

        self.index.build_index(self.lifted_pts)
        return self

    def query(self, q, k):
        if self.lifted_pts is None:
            raise RuntimeError("Index is not built. Call build(pts) first.")

        n = int(self.lifted_pts.shape[0])
        if k <= 0:
            raise ValueError("k must be >= 1")
        if k > n:
            k = n

        # The underlying stripe logic in EpsHierIndex treats `k` like a rank index.
        # This wrapper's API is the conventional 1-based "k-th nearest neighbor",
        # so convert to a 0-based rank for exact thresholds.
        k_rank = int(k - 1)

        q = np.asarray(q)

        if self.norm == "l2":
            if q.ndim != 1:
                raise ValueError(f"q must be 1D. Got shape={q.shape}")
            lifted_q = np.zeros((q.shape[0] + 1,), dtype=q.dtype)
            lifted_q[: q.shape[0]] = -2 * q
            lifted_q[q.shape[0]] = 1

            lifted_q = -lifted_q

            norm = np.linalg.norm(lifted_q)
            if norm == 0:
                raise ValueError("Lifted query vector has zero norm.")
            lifted_q = lifted_q / norm
            return self.index.query(lifted_q, k_rank)
        else:  # cosine
            if q.ndim != 1:
                raise ValueError(f"q must be 1D. Got shape={q.shape}")
            norm = np.linalg.norm(q)
            if norm == 0:
                raise ValueError("Query vector has zero norm.")
            normalized_q = q / norm
            return self.index.query(normalized_q, k_rank)
