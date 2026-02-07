"""
Data loader for standard ANN benchmark datasets from ann-benchmarks.com
"""

import os
import h5py
import numpy as np
import urllib.request
from pathlib import Path
from typing import Optional, Tuple


# Dataset URLs from ann-benchmarks
DATASET_URLS = {
    "glove-100-angular": "http://ann-benchmarks.com/glove-100-angular.hdf5",
    "glove-25-angular": "http://ann-benchmarks.com/glove-25-angular.hdf5",
    "nytimes-256-angular": "http://ann-benchmarks.com/nytimes-256-angular.hdf5",
    "fashion-mnist-784-euclidean": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
    "gist-960-euclidean": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
    "sift-128-euclidean": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
}


def get_data_dir() -> Path:
    """Get the data directory path (git-ignored)."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download_dataset(dataset_name: str) -> Path:
    """
    Download a dataset from ann-benchmarks.com if not already present.

    Args:
        dataset_name: Name of the dataset (e.g., 'glove-100-angular')

    Returns:
        Path to the downloaded HDF5 file
    """
    if dataset_name not in DATASET_URLS:
        available = ", ".join(DATASET_URLS.keys())
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

    data_dir = get_data_dir()
    file_path = data_dir / f"{dataset_name}.hdf5"

    if file_path.exists():
        print(f"Dataset {dataset_name} already exists at {file_path}")
        return file_path

    url = DATASET_URLS[dataset_name]
    print(f"Downloading {dataset_name} from {url}...")

    try:
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded to {file_path}")
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise Exception(f"Failed to download {dataset_name}: {e}")

    return file_path


def load_dataset(
    dataset_name: str,
    n: Optional[int] = None,
    return_queries: bool = False,
    seed: int = 42,
    normalize_angular: bool = True,
) -> Tuple[np.ndarray, ...]:
    """
    Load an ANN benchmark dataset.

    Args:
        dataset_name: Name of the dataset. Options:
            - 'glove-100-angular': GloVe 100d word embeddings (angular distance)
            - 'glove-25-angular': GloVe 25d word embeddings (angular distance)
            - 'nytimes-256-angular': NY Times article embeddings (angular distance)
            - 'fashion-mnist-784-euclidean': Fashion-MNIST images (L2 distance)
            - 'gist-960-euclidean': GIST image descriptors (L2 distance)
            - 'sift-128-euclidean': SIFT image descriptors (L2 distance)
        n: Number of points to load. If None, loads all points.
           If specified, takes a random sample.
        return_queries: If True, also returns the query set and ground truth neighbors
        seed: Random seed for sampling (if n is specified)

    Returns:
        If return_queries is False:
            train_data: np.ndarray of shape (n_samples, n_dims)
        If return_queries is True:
            (train_data, queries, ground_truth): Tuple of arrays
            - train_data: Training vectors
            - queries: Query vectors
            - ground_truth: Indices of ground truth neighbors for each query
    """
    # Download dataset if needed
    file_path = download_dataset(dataset_name)

    def _drop_zero_norm_rows(
        x: np.ndarray, *, eps: float = 1e-12
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape={x.shape}")
        norms = np.linalg.norm(x, axis=1)
        mask = norms > eps
        return x[mask], mask

    def _row_normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array to normalize, got shape={x.shape}")
        x = x.astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return x / norms

    def _is_angular_dataset(h5: h5py.File, name: str) -> bool:
        # Prefer the dataset's explicit metadata when present.
        if "distance" in h5:
            dist = h5["distance"][()]
            if isinstance(dist, (bytes, np.bytes_)):
                dist = dist.decode("utf-8", errors="ignore")
            if isinstance(dist, str):
                return dist.strip().lower() == "angular"
        # Fallback to the naming convention used by ann-benchmarks.
        return "angular" in name.lower()

    # Load data from HDF5 file
    with h5py.File(file_path, "r") as f:
        train_full = np.array(f["train"])
        train = train_full

        # Track original indices when we sample/filter, so we can remap neighbors.
        train_old_indices = np.arange(train.shape[0])

        # Sample if n is specified
        if n is not None and n < len(train):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(train), size=n, replace=False)
            train = train[indices]
            train_old_indices = train_old_indices[indices]

        # NYTimes contains some zero-norm vectors; drop them to avoid undefined
        # cosine similarity and NaNs in downstream metrics.
        if dataset_name.lower() == "nytimes-256-angular":
            train, train_mask = _drop_zero_norm_rows(train)
            train_old_indices = train_old_indices[train_mask]

        if normalize_angular and _is_angular_dataset(f, dataset_name):
            train = _row_normalize(train)

        if return_queries:
            if dataset_name.lower() == "nytimes-256-angular" and n is not None:
                raise ValueError(
                    "For nytimes-256-angular, return_queries=True is only supported with n=None "
                    "because the provided ground-truth neighbors are indexed over the full train set. "
                    "Use n=None or set return_queries=False and sample train yourself."
                )
            test = np.array(f["test"])
            neighbors = np.array(f["neighbors"])

            if dataset_name.lower() == "nytimes-256-angular":
                # Drop zero-norm queries and keep neighbors in sync.
                test, test_mask = _drop_zero_norm_rows(test)
                neighbors = neighbors[test_mask]

            if normalize_angular and _is_angular_dataset(f, dataset_name):
                test = _row_normalize(test)

            if dataset_name.lower() == "nytimes-256-angular":
                # Remap neighbor indices from original train space -> filtered train.
                old_to_new = np.full((train_full.shape[0],), -1, dtype=np.int64)
                old_to_new[train_old_indices.astype(np.int64, copy=False)] = np.arange(
                    train.shape[0], dtype=np.int64
                )

                mapped = old_to_new[neighbors.astype(np.int64, copy=False)]
                # Keep order, drop missing (-1), and pad to original width.
                width = mapped.shape[1]
                remapped = np.full_like(mapped, -1)
                for i in range(mapped.shape[0]):
                    row = mapped[i]
                    valid = row[row >= 0]
                    if valid.size:
                        remapped[i, : min(width, valid.size)] = valid[:width]
                neighbors = remapped

            return train, test, neighbors

        return train


def get_dataset_info(dataset_name: str) -> dict:
    """
    Get information about a dataset without loading it.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Dictionary with dataset information (size, dimensions, distance metric)
    """
    file_path = download_dataset(dataset_name)

    with h5py.File(file_path, "r") as f:
        train = f["train"]
        test = f["test"]

        info = {
            "name": dataset_name,
            "train_size": train.shape[0],
            "test_size": test.shape[0],
            "dimensions": train.shape[1],
            "distance": (
                f["distance"][()].decode("utf-8")
                if "distance" in f
                else ("angular" if "angular" in dataset_name else "euclidean")
            ),
        }

    return info


if __name__ == "__main__":
    # Example usage
    print("Available datasets:")
    for name in DATASET_URLS.keys():
        print(f"  - {name}")

    # Load a small sample
    print("\nLoading 1000 points from glove-25-angular...")
    data = load_dataset("glove-25-angular", n=1000)
    print(f"Loaded data shape: {data.shape}")

    # Get dataset info
    print("\nDataset info:")
    info = get_dataset_info("glove-25-angular")
    for key, value in info.items():
        print(f"  {key}: {value}")
