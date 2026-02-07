import numpy as np
import time
import pandas as pd
import sys
import pickle
import psutil
import os

sys.path.append("../..")

from knn.methods.epshier import EpsHierANNIndex, EpsHierConfig
from knn.datasets.data_loader import get_dataset_info, load_dataset
from knn.methods.faiss import *


def get_object_size_mb(obj):
    """Get the size of a pickled object in MB"""
    try:
        serialized = pickle.dumps(obj)
        return len(serialized) / (1024 * 1024)
    except Exception as e:
        print(f"Warning: Could not pickle object - {e}")
        return -1


def get_memory_usage_mb():
    """Get current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


best_config = {
    "glove-100-angular": {
        "eps": 1 / 2**5,
        "constant": 8,
        "metric": "cosine",
    },
    "glove-25-angular": {
        "eps": 1 / 2**5,
        "constant": 32,
        "metric": "cosine",
    },
    "fashion-mnist-784-euclidean": {
        "eps": 1 / 2**5,
        "constant": 0.5,
        "metric": "l2",
    },
    "gist-960-euclidean": {
        "eps": 1 / 2**5,
        "constant": 0.5,
        "metric": "l2",
    },
    "sift-128-euclidean": {
        "eps": 1 / 2**6,
        "constant": 2,
        "metric": "l2",
    },
    "nytimes-256-angular": {
        "eps": 1 / 2**5,
        "constant": 8,
        "metric": "cosine",
    },
}


def benchmark_index(dataset_name, n=100000):
    """Benchmark all indexes on a given dataset"""
    print(f"\n{'='*60}")
    print(f"Benchmarking dataset: {dataset_name}")
    print(f"{'='*60}\n")

    # Load dataset
    data = load_dataset(dataset_name=dataset_name, n=n).astype(np.float32)
    print(f"Data shape: {data.shape}")

    metric = best_config[dataset_name]["metric"]
    print(f"Metric: {metric}\n")

    results = {
        "dataset": [],
        "method": [],
        "build_time_sec": [],
        "space_usage_mb": [],
        "n_points": [],
        "n_dims": [],
    }

    # EpsHier configuration
    num_levels = 10
    branching_factor = 10
    eps = best_config[dataset_name]["eps"]
    constant = best_config[dataset_name]["constant"] * 1 / 2**10

    # Benchmark EpsHier
    print("Building EpsHier index...")
    mem_before = get_memory_usage_mb()
    start = time.time()

    config = EpsHierConfig(
        num_levels=num_levels,
        branching_factor=branching_factor,
        eps=eps,
        norm=metric,
        eps_sample_constant=constant,
    )
    eps_index = EpsHierANNIndex(config, verbose=False).build_index(data)

    build_time = time.time() - start
    mem_after = get_memory_usage_mb()
    mem_delta = mem_after - mem_before
    obj_size = get_object_size_mb(eps_index)

    results["dataset"].append(dataset_name)
    results["method"].append("EpsHier")
    results["build_time_sec"].append(build_time)
    results["space_usage_mb"].append(obj_size if obj_size > 0 else mem_delta)
    results["n_points"].append(n)
    results["n_dims"].append(data.shape[1])

    print(f"  Build time: {build_time:.3f}s")
    print(f"  Space usage: {results['space_usage_mb'][-1]:.2f} MB\n")

    # FAISS methods
    faiss_methods = [
        ("HNSW", lambda: FaissHNSWIndex().build(data, metric)),
        ("HNSW-PQ", lambda: FaissHNSWPQIndex().build(data, metric)),
        ("IVFPQ", lambda: FaissIVFPQIndex().build(data, metric)),
        ("PQ-Flat", lambda: FaissPQFlatIndex().build(data, metric)),
        ("LSH", lambda: FaissLSHIndex().build(data, metric)),
        ("IVFFlat", lambda: FaissIVFFlatIndex().build(data, metric)),
    ]

    for method_name, build_func in faiss_methods:
        print(f"Building {method_name} index...")
        mem_before = get_memory_usage_mb()
        start = time.time()

        try:
            index = build_func()
            build_time = time.time() - start
            mem_after = get_memory_usage_mb()
            mem_delta = mem_after - mem_before
            obj_size = get_object_size_mb(index)

            results["dataset"].append(dataset_name)
            results["method"].append(method_name)
            results["build_time_sec"].append(build_time)
            results["space_usage_mb"].append(obj_size if obj_size > 0 else mem_delta)
            results["n_points"].append(n)
            results["n_dims"].append(data.shape[1])

            print(f"  Build time: {build_time:.3f}s")
            print(f"  Space usage: {results['space_usage_mb'][-1]:.2f} MB\n")

        except Exception as e:
            print(f"  Error building {method_name}: {e}\n")
            continue

    return pd.DataFrame(results)


def main():
    # Datasets to benchmark
    datasets = [
        "glove-100-angular",
        "sift-128-euclidean",
        "fashion-mnist-784-euclidean",
        "gist-960-euclidean",
    ]

    all_results = []

    for dataset in datasets:
        try:
            df = benchmark_index(dataset, n=100000)
            all_results.append(df)
        except Exception as e:
            print(f"Error benchmarking {dataset}: {e}")
            continue

    # Combine and save results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        output_file = "../artifacts/space_time_benchmark.csv"
        final_df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}\n")
        print(final_df.to_string(index=False))
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
