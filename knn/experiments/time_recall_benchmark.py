# %%
import numpy as np
from tqdm import tqdm
import time
import pandas as pd

import sys

sys.path.append("../..")

from knn.methods.epshier import EpsHierANNIndex, EpsHierIndex, EpsHierConfig
from knn.datasets.data_loader import get_dataset_info, load_dataset
from knn.methods.faiss import *
from knn.experiments.utils import brute_force
from knn.methods.baseline import quickselect, calc_score

# %%
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
}

# %%
for dataset in best_config.keys():
    info = get_dataset_info(dataset)
    print(f"Dataset: {dataset}")
    print(info)

# %%
tmp = pd.read_csv("../artifacts/time_benchmark_glove-100-angular_v3.csv")
tmp[(tmp["method_name"] == "IVFFlat") & (tmp["k"] == 591757)]["output_size"].mean()
# tmp["k"].value_counts()

# %%
# config
dataset_name = "glove-100-angular"
df = load_dataset(dataset_name=dataset_name)
print(df.shape)
metric = best_config[dataset_name]["metric"]  # l2, cosine
n = min(df.shape[0], 100000)
# k = n // 2

# eps hier
num_levels = 10
branching_factor = 10
eps = best_config[dataset_name]["eps"]
constant = best_config[dataset_name]["constant"] * 1 / 2**10

# benchmarking
num_queries = 100

# %% [markdown]
# ### Indexing

# %%
data = load_dataset(dataset_name=dataset_name, n=n).astype(np.float32)

# eps hier
config = EpsHierConfig(
    num_levels=num_levels,
    branching_factor=branching_factor,
    eps=eps,
    norm=metric,
    eps_sample_constant=constant,
)
eps_index = EpsHierANNIndex(config, verbose=True).build_index(data)

# FAISS
print("Building HNSW indexes...")
hnsw = FaissHNSWIndex().build(data, metric)
print("Building HNSW-PQ indexes...")
hnsw_pq = FaissHNSWPQIndex().build(data, metric)
print("Building IVFPQ indexes...")
ivfpq = FaissIVFPQIndex().build(data, metric)
print("Building PQ-Flat indexes...")
pq = FaissPQFlatIndex().build(data, metric)
print("Building LSH indexes...")
lsh = FaissLSHIndex().build(data, metric)
print("Building IVFFlat indexes...")
ivf = FaissIVFFlatIndex().build(data, metric)

# %% [markdown]
# ### Querying

# %%
# def brute_force(data, k, q, metric):
#     if metric == "l2":
#         # calculate all scores O(n)
#         scores = []
#         for i, point in enumerate(data):
#             scores.append((i, np.linalg.norm(point - q)))
#         # sort scores O(n log n)
#         scores.sort(key=lambda x: x[1], reverse=False)  # closest to furthest
#         # get k-th index
#         idx = scores[k - 1][0]
#         return idx

#     if metric == "cosine":
#         # calculate all scores O(n)
#         scores = []
#         for i, point in enumerate(data):
#             scores.append(
#                 (i, np.dot(point, q) / (np.linalg.norm(point) * np.linalg.norm(q)))
#             )
#         # sort scores O(n log n)
#         scores.sort(key=lambda x: x[1], reverse=True)  # closest to furthest
#         # get k-th index
#         idx = scores[k - 1][0]

#         return idx


def get_recall(truth, output):
    return 1.0 if truth in output else 0.0


def faiss_output_postprocess(output, size, data, query):
    scores = []
    query_norm = np.linalg.norm(query)
    for idx in output:
        # drop invalid indices
        if idx == -1:
            continue

        point = data[idx]
        if metric == "l2":
            scores.append((idx, np.linalg.norm(point - query)))
        elif metric == "cosine":
            scores.append(
                (idx, np.dot(point, query) / (np.linalg.norm(point) * query_norm))
            )
        else:
            raise ValueError(f"Unknown metric: {metric}")

    scores.sort(key=lambda x: x[1], reverse=(metric == "cosine"))
    crop_size = min(int(size), len(scores))
    cropped = scores[:crop_size]

    return [idx for idx, _ in cropped]

# %% [markdown]
# ### Benchmark

# %%
faiss_methods = {
    # "EpsHier": lambda data, query, k, metric: eps_index.query(query, k),
    "HNSW": lambda data, query, k, metric: hnsw.knn(query, k),
    "IVFPQ": lambda data, query, k, metric: ivfpq.knn(query, k),
    "PQ": lambda data, query, k, metric: pq.knn(query, k),
    "HNSWPQ": lambda data, query, k, metric: hnsw_pq.knn(query, k),
    "LSH": lambda data, query, k, metric: lsh.knn(query, k),
    "IVFFlat": lambda data, query, k, metric: ivf.knn(query, k),
}

import math

logn = math.floor(math.log2(n))

ks = [n // (2**i) for i in range(logn, 0, -1)]

ks = [10, *ks, 3 * n // 4]

results = {"times": [], "recalls": [], "method_name": [], "k": [], "output_size": []}

# warm up for numba
for _ in range(5):
    _ = eps_index.query(data[0], 10)

for i in tqdm(range(num_queries)):
    query = data[np.random.choice(n, size=1, replace=False)][0]

    for k in ks:
        # Brute force
        start = time.time()
        # bf_idx = brute_force(data, k, query, metric)
        bf_idx = quickselect(data, k - 1, query, metric)
        end = time.time()
        bf_time = end - start

        results["times"].append(bf_time)
        results["recalls"].append(1)  # brute force always has recall 1
        results["method_name"].append("Quick Select")
        results["k"].append(k)
        results["output_size"].append(1)

        # EpsHier
        start = time.time()
        eps_idx = eps_index.query(query, k)
        end = time.time()

        results["times"].append(end - start)
        results["recalls"].append(get_recall(bf_idx, eps_idx))
        results["method_name"].append("EpsHier")
        results["k"].append(k)
        results["output_size"].append(len(eps_idx))

        eps_hier_output_size = len(eps_idx)

        for method_name, func in faiss_methods.items():
            start = time.time()
            idxs = func(data, query, k, metric)
            idxs = faiss_output_postprocess(idxs, eps_hier_output_size, data, query)
            end = time.time()

            results["recalls"].append(get_recall(bf_idx, idxs))
            results["times"].append(end - start)
            results["method_name"].append(method_name)
            results["k"].append(k)
            results["output_size"].append(len(idxs))

    # store results
    if i % 10 == 0 and i > 0:
        pd.DataFrame(results).to_csv(
            f"../artifacts/time_benchmark_{dataset_name}_v3.csv", index=False
        )

pd.DataFrame(results).to_csv(
    f"../artifacts/time_benchmark_{dataset_name}_v3.csv", index=False
)


