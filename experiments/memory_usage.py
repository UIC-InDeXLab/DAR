# %%
import sys

sys.path.append("..")

import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np
import time
from pympler import asizeof

from methods.kth.epssample import *
from methods.kth.fagin import *
from methods.kth.threshold import *
from methods.utils import *
from methods.range_search.hierarchy import *
from methods.range_search.hierarchy import *
from methods.range_search.kdtree import *
from methods.range_search.rtree_impl import *
from methods.range_search.partition_tree import *


# %%
def build_indices(n, dim):
    points = np.random.zipf(1.5, size=(n, dim))

    # Hierarchical
    start = time.time()
    hie = HierarchicalIndex(points=points, decay=4)
    hie.build_index()
    hie.find_coverage()
    hie.find_neighbor_stats()
    t1 = time.time() - start

    # KDTree
    start = time.time()
    kdtree = KDTree(points=points)
    t2 = time.time() - start

    # RTree
    start = time.time()
    rtree = RTree(dimensions=dim)
    rtree.insert_points(points)
    t3 = time.time() - start

    # Partition Tree
    start = time.time()
    partition_tree = PartitionTree(Point.from_numpy(points))
    t4 = time.time() - start

    return {
        "hierarchical": (hie, t1),
        "kdtree": (kdtree, t2),
        "rtree": (rtree, t3),
        "partition_tree": (partition_tree, t4),
    }, points


# %%
ns = [10_000, 20_000, 50_000, 100_000, 500_000, 1_000_000]
dims = [8]

results = {}

get_list_size = lambda x: sys.getsizeof(x) + sum(sys.getsizeof(i) for i in x)
get_array_size = lambda x: sys.getsizeof(x) + x.nbytes

for n in ns:
    print(f"Processing n={n} / {ns}")
    fns, points = build_indices(n, dims[0])

    for method, fn in fns.items():
        ind = fn[0]
        t = fn[1]
        print(f"Method: {method}")
        results.setdefault(method, []).append(get_size_recursive(ind))
        results.setdefault(f"{method}_time", []).append(t)

    results.setdefault("n", []).append(n)
    results.setdefault("dim", []).append(dims[0])

results = pd.DataFrame(results)

# %%
# results.to_csv("assets/kth_results_prep_memory_sizes.csv", index=False)
results.head()

# %% [markdown]
# Visualize

# %%
df = results.copy()
df.head()

# %%

df = results.copy()

# Plot memory usage vs dataset size
plt.figure(figsize=(8, 6))
plt.plot(df["n"], df["hierarchical"], marker="o", label="Hierarchical", color="blue")
plt.plot(df["n"], df["kdtree"], marker="o", label="KD-Tree", color="red")
plt.plot(df["n"], df["rtree"], marker="o", label="R-Tree", color="orange")
plt.plot(
    df["n"], df["partition_tree"], marker="o", label="Partition Tree", color="purple"
)

plt.xlabel("Dataset Size (n)", fontsize=18)
plt.ylabel("Memory Usage (bytes)", fontsize=18)
plt.title("Memory Usage vs Dataset Size", fontsize=22)
plt.legend(fontsize=19)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.grid(True)
plt.tight_layout()
# plt.savefig("assets/memory_usage_vs_dataset_size.png")
plt.show()


# %%
df = results.copy()

# Plot memory usage vs dataset size
plt.figure(figsize=(8, 6))
plt.plot(
    df["n"], df["hierarchical_time"], marker="o", label="Hierarchical", color="blue"
)
plt.plot(df["n"], df["kdtree_time"], marker="o", label="KD-Tree", color="red")
plt.plot(df["n"], df["rtree_time"], marker="o", label="R-Tree", color="orange")
plt.plot(
    df["n"],
    df["partition_tree_time"],
    marker="o",
    label="Partition Tree",
    color="purple",
)

plt.xlabel("Dataset Size (n)", fontsize=18)
plt.ylabel("Preprocessing Time (s)", fontsize=18)
plt.title("Preprocessing Time vs Dataset Size", fontsize=22)
plt.legend(fontsize=19)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.xscale("log", base=2)
plt.yscale("log", base=2)
# plt.savefig("assets/preprocessing_time_vs_dataset_size.png")
plt.show()
