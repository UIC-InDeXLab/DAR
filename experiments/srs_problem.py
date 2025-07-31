# %%
import sys

sys.path.append("..")

import numpy as np
import time
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import gc

from methods.range_search.hierarchy import *
from methods.range_search.kdtree import *
from methods.range_search.rtree_impl import *
from methods.range_search.partition_tree import *
from ranges.stripe_range import StripeRange

# %% [markdown]
# Helpers


# %%
def build_indices(n, dim, points=None):
    if points is None:
        points = np.random.zipf(1.5, size=(n, dim))

    n = points.shape[0]
    dim = points.shape[1]

    # Hierarchical
    print("Indexing hierarchical...")
    hie = HierarchicalIndex(points=points, decay=4)
    hie.build_index()
    hie.find_coverage()
    hie.find_neighbor_stats()

    # KDTree
    # print("Indexing KDTree...")
    # kdtree = KDTree(points=points)

    # RTree
    # print("Indexing RTree...")
    # rtree = RTree(dimensions=dim)
    # rtree.insert_points(points)

    # Partition Tree
    print("Indexing Partition Tree...")
    partition_tree = PartitionTree(Point.from_numpy(points))

    return {
        "hierarchical": lambda q: hie.query(q),
        # "kdtree": lambda q: kdtree.query(q),
        # "rtree": lambda q: rtree.query(q),
        "partition_tree": lambda q: partition_tree.halfspace_query(
            q.normal_vector.tolist(), q.start_dot, q.end_dot
        ),
    }, points


# %%
def sample_query(points, width):
    return StripeRange.sample_stripe(points, r=width, tolerance=0.001)


# %%
def get_recall(results, gt):
    return len(results) / len(gt) if len(gt) > 0 else 1.0


# %%
def round_to_nearest_power_fraction(arr):
    """
    Round each element in arr to the nearest 1 / (2^i) for i = 1, 2, ..., such that 1/(2^i) < 1.
    Assumes all elements in arr are in (0,1).
    """
    powers = np.array([1 / (2**i) for i in range(1, 20)])  # Adjust range if needed
    arr = np.asarray(arr)

    def nearest_fraction(x):
        diffs = np.abs(powers - x)
        return powers[np.argmin(diffs)]

    return np.vectorize(nearest_fraction)(arr)


# %% [markdown]
# Different dimensions

# %%
n = 10000
dims = [2, 4, 8, 32, 64, 128]
repeat = 5
widths = [
    1 / 2,
    1 / 4,
    1 / 8,
    1 / 16,
    1 / 32,
    1 / 64,
    1 / 128,
    1 / 256,
    1 / 512,
]

results = {}

for d in dims:
    fns, points = build_indices(n, d)

    for width in tqdm(widths, desc=f"Width {d}D"):
        query = sample_query(points, width=width)

        # linear search
        start_time = time.time()
        for _ in range(repeat):
            gt = linear_search(points, query)
        elapsed_time = time.time() - start_time
        results.setdefault("linear_time", []).append(elapsed_time / repeat)

        for method, fn in fns.items():
            print("Running method:", method)
            if method == "linear":
                continue
            start_time = time.time()
            for _ in range(repeat):
                res = fn(query)
            elapsed_time = time.time() - start_time
            results.setdefault(f"{method}_time", []).append(elapsed_time / repeat)
            recall = get_recall(res, gt)
            results.setdefault(f"{method}_recall", []).append(recall)

        results.setdefault("widths", []).append(width)
        results.setdefault("dimensions", []).append(d)
        results.setdefault("n", []).append(n)

# %%
results = pd.DataFrame(results)
results["frac"] = round_to_nearest_power_fraction(results["widths"])
# results.to_csv("assets/partition_tree_range_search_time_vs_dim.csv", index=False)

# %% [markdown]
# Visualization

# %%
results = pd.read_csv("assets/range_search_dim_2.csv")
part_tree_results = pd.read_csv("assets/partition_tree_range_search_time_vs_dim.csv")

# %%
results["dimensions"].value_counts(), part_tree_results["dimensions"].value_counts()

# %%
merged_df = pd.merge(
    results,
    part_tree_results[
        ["n", "dimensions", "widths", "partition_tree_time", "partition_tree_recall"]
    ],
    on=["n", "dimensions", "widths"],
    how="left",  # or 'inner' if you only want matches
)
merged_df.head()
results = merged_df

# %%
d = 32
n = 10000
df = results[results["dimensions"] == d]

# Group by frac and compute mean times
grouped = df.groupby("frac").mean(numeric_only=True)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(
    grouped.index,
    grouped["hierarchical_time"],
    marker="x",
    label="Hierarchical Index",
    color="blue",
)
# plt.plot(grouped.index, grouped["rtree_time"], marker="o", label="R-Tree", color='orange')
plt.plot(
    grouped.index,
    grouped["linear_time"],
    marker="s",
    label="Linear Time",
    color="green",
)
plt.plot(
    grouped.index,
    grouped["partition_tree_time"],
    marker="s",
    label="Partition Tree",
    color="purple",
)
# plt.plot(grouped.index, grouped["kdtree_time"], marker="^", label="KD-Tree", color='red')

# Formatting
plt.xlabel("Stripe Range Width (Fraction)", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title(rf"Query Time vs. Stripe Width ($d={d}, n={n}$)", fontsize=22)
plt.legend(fontsize=19, framealpha=0.7)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig(f'assets/range_search_dim_{d}.png')
plt.show()

# %% [markdown]
# Different sizes of input:

# %%
# ns = [10000, 20000, 40000, 80000]#, 160000]
ns = [1_000_000]
d = 4
repeat = 5
widths = [
    1 / 2,
    1 / 4,
    1 / 8,
    1 / 16,
    1 / 32,
    1 / 64,
    1 / 128,
    1 / 256,
    1 / 512,
]

results = {}

for n in ns:
    fns, points = build_indices(n, d)

    for width in tqdm(widths, desc=f"Width {n} size"):
        query = sample_query(points, width=width)

        # linear search
        start_time = time.time()
        for _ in range(repeat):
            gt = linear_search(points, query)
        elapsed_time = time.time() - start_time
        results.setdefault("linear_time", []).append(elapsed_time / repeat)

        for method, fn in fns.items():
            print("Running method:", method)
            if method == "linear":
                continue
            start_time = time.time()
            for _ in range(repeat):
                res = fn(query)
            elapsed_time = time.time() - start_time
            results.setdefault(f"{method}_time", []).append(elapsed_time / repeat)
            recall = get_recall(res, gt)
            results.setdefault(f"{method}_recall", []).append(recall)

        results.setdefault("widths", []).append(width)
        results.setdefault("dimensions", []).append(d)
        results.setdefault("n", []).append(n)

# %%
results = pd.DataFrame(results)
# results.to_csv("assets/partition_tree_range_search_time_vs_size(dim=4).csv", index=False)

# %% [markdown]
# visualize
#

# %%
results = pd.concat(
    [
        pd.read_csv("assets/range_search_size_1.csv"),
        pd.read_csv("assets/range_search_size_2.csv"),
        pd.read_csv("assets/range_search_size_3.csv"),
        # pd.read_csv("assets/range_search_size_dim_128_1.csv"),
        # pd.read_csv("assets/range_search_size_dim_128_2.csv")
    ],
    ignore_index=True,
)

# part_results = pd.read_csv("assets/partition_tree_range_search_time_vs_size(dim=128).csv")
part_results = pd.read_csv("assets/partition_tree_range_search_time_vs_size(dim=4).csv")

results.head()

# %%
merged_df = pd.merge(
    results,
    part_results[
        ["n", "dimensions", "widths", "partition_tree_time", "partition_tree_recall"]
    ],
    on=["n", "dimensions", "widths"],
    how="left",  # or 'inner' if you only want matches
)
merged_df.head()
results = merged_df
results = pd.concat([results, pd.read_csv("assets/million_d=4.csv")], ignore_index=True)
# results = pd.concat([results, pd.read_csv("assets/million_d=128.csv")], ignore_index=True)

# %%
df = results
# Group by n and compute mean times
grouped_n = df.groupby("n").mean(numeric_only=True)

grouped_n = grouped_n.drop(grouped_n.index[[2, 4]])

# Plot
plt.figure(figsize=(8, 6))
plt.plot(
    grouped_n.index,
    grouped_n["hierarchical_time"],
    marker="x",
    label="Hierarchical Index",
    color="blue",
)
plt.plot(
    grouped_n.index, grouped_n["rtree_time"], marker="o", label="R-Tree", color="orange"
)
plt.plot(
    grouped_n.index,
    grouped_n["linear_time"],
    marker="s",
    label="Linear Time",
    color="green",
)
plt.plot(
    grouped_n.index,
    grouped_n["partition_tree_time"],
    marker="s",
    label="Partition Tree Time",
    color="purple",
)
plt.plot(
    grouped_n.index, grouped_n["kdtree_time"], marker="^", label="KD-Tree", color="red"
)

# Formatting
plt.xlabel(r"Size of Dataset ($n$)", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title(rf"Query Time vs. Dataset Size ($d={results['dimensions'][0]}$)", fontsize=22)
plt.legend(fontsize=17, framealpha=0.7)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig(f'assets/range_search_size_dim_{results["dimensions"][0]}.png')
plt.show()

# %%
# dim = 128

df = results
df = df[df["partition_tree_time"].notna()]

# Group by n and compute mean times
grouped_n = df.groupby("n").mean(numeric_only=True)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(
    grouped_n.index,
    grouped_n["hierarchical_time"],
    marker="x",
    label="Hierarchical Index",
    color="blue",
)
plt.plot(
    grouped_n.index,
    grouped_n["linear_time"],
    marker="s",
    label="Linear Time",
    color="green",
)
plt.plot(
    grouped_n.index,
    grouped_n["partition_tree_time"],
    marker="s",
    label="Partition Tree Time",
    color="purple",
)

# Formatting
plt.xlabel("Number of Points (n)", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title(
    f"Query Times vs Input Size (dimension={results['dimensions'][0]})", fontsize=22
)
plt.legend(fontsize=17)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig(f'assets/range_search_size_dim_{results["dimensions"][0]}.png')
plt.show()

# %% [markdown]
# Real Dataset: US Used Cars

# %%
import kagglehub

dataset = kagglehub.dataset_download("ananaymital/us-used-cars-dataset")

# %%
df = pd.read_csv(f"{dataset}/used_cars_data.csv")
df.head()

# %%
df.columns

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Step 1: Drop unwanted columns
drop_cols = ["vin", "listing_id", "main_picture_url", "sp_id", "trimId"]
data = df.drop(columns=drop_cols)
data = data.fillna(0)

# Step 2: Identify column types
cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Convert all categorical values to strings and fill missing with 'Unknown'
data[cat_cols] = data[cat_cols].astype(str).fillna("Unknown")

# Fill numeric missing values with column mean
for col in num_cols:
    data[col] = data[col].fillna(data[col].mean())

# Step 3: Define transformer with OrdinalEncoder + MinMaxScaler
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(-1, 1)), num_cols),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_cols,
        ),
    ]
)

# Step 4: Transform entire dataset
data = preprocessor.fit_transform(data)

# Optional: feature names
feature_names = preprocessor.get_feature_names_out()

# Cleanup
del df
gc.collect()

# Shape of final data
data.shape

# %%
repeat = 5
widths = [
    1 / 2,
    1 / 4,
    1 / 8,
    1 / 16,
    1 / 32,
    1 / 64,
    1 / 128,
    1 / 256,
    1 / 512,
]

results = {}


fns, points = build_indices(-1, -1, points=data)

for width in tqdm(widths):
    query = sample_query(points, width=width)

    # linear search
    start_time = time.time()
    for _ in range(repeat):
        gt = linear_search(points, query)
    elapsed_time = time.time() - start_time
    results.setdefault("linear_time", []).append(elapsed_time / repeat)

    for method, fn in fns.items():
        print("Running method:", method)
        if method == "linear":
            continue
        start_time = time.time()
        for _ in range(repeat):
            res = fn(query)
        elapsed_time = time.time() - start_time
        results.setdefault(f"{method}_time", []).append(elapsed_time / repeat)
        recall = get_recall(res, gt)
        results.setdefault(f"{method}_recall", []).append(recall)

    results.setdefault("widths", []).append(width)

results = pd.DataFrame(results)

# %%
# results.to_csv("assets/range_search_used_cars.csv", index=False)

# %% [markdown]
# Visualization

# %%
df_hier = pd.read_csv("assets/hierarchical_used_cars.csv")
df_part = pd.read_csv("assets/partition_tree_used_cars.csv")

df_hier.head()

# %%
df_part.head()

# %%
results = df_part.merge(df_hier, on=["widths"], suffixes=("_part", "_hier"))
results["frac"] = round_to_nearest_power_fraction(results["widths"])
results.head()

# %%
# Group by frac and compute mean times
df = results.copy()
grouped = df.groupby("frac").mean(numeric_only=True)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(
    grouped.index,
    grouped["hierarchical_time"],
    marker="x",
    label="Hierarchical Index",
    color="blue",
)
# plt.plot(grouped.index, grouped["rtree_time"], marker="o", label="R-Tree", color='orange')
plt.plot(
    grouped.index,
    grouped["linear_time_hier"],
    marker="s",
    label="Linear Time",
    color="green",
)
plt.plot(
    grouped.index,
    grouped["partition_tree_time"],
    marker="s",
    label="Partition Tree",
    color="purple",
)
# plt.plot(grouped.index, grouped["kdtree_time"], marker="^", label="KD-Tree", color='red')

# Formatting
plt.xlabel("Stripe Range Width (fraction)", fontsize=18)
plt.ylabel("Time (s)", fontsize=18)
plt.title(f"Query Time (US Used Cars Dataset)", fontsize=22)
plt.legend(fontsize=19)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig(f'assets/range_search_us_used_cars.png')
plt.show()

# %% [markdown]
# FIFA Dataset

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("stefanoleone992/fifa-23-complete-player-dataset")

# %%
df = pd.read_csv(f"{path}/male_teams.csv")
df.shape

# %%
df.columns

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Step 1: Drop unwanted columns
drop_cols = [
    "team_id",
    "team_url",
    "fifa_version",
    "fifa_update",
    "league_id",
    "coach_id",
]
data = df.drop(columns=drop_cols)
data = data.fillna(0)

# Step 2: Identify column types
cat_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# Convert all categorical values to strings and fill missing with 'Unknown'
data[cat_cols] = data[cat_cols].astype(str).fillna("Unknown")

# Fill numeric missing values with column mean
for col in num_cols:
    data[col] = data[col].fillna(data[col].mean())

# Step 3: Define transformer with OrdinalEncoder + MinMaxScaler
preprocessor = ColumnTransformer(
    transformers=[
        ("num", MinMaxScaler(feature_range=(-1, 1)), num_cols),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_cols,
        ),
    ]
)

# Step 4: Transform entire dataset
data = preprocessor.fit_transform(data)

# Optional: feature names
feature_names = preprocessor.get_feature_names_out()

# Cleanup
del df
gc.collect()

# Shape of final data
data.shape

# %%
repeat = 5
widths = [
    1 / 2,
    1 / 4,
    1 / 8,
    1 / 16,
    1 / 32,
    1 / 64,
    1 / 128,
    1 / 256,
    1 / 512,
]

results = {}


fns, points = build_indices(-1, -1, points=data)

for width in tqdm(widths):
    query = sample_query(points, width=width)

    # linear search
    start_time = time.time()
    for _ in range(repeat):
        gt = linear_search(points, query)
    elapsed_time = time.time() - start_time
    results.setdefault("linear_time", []).append(elapsed_time / repeat)

    for method, fn in fns.items():
        print("Running method:", method)
        if method == "linear":
            continue
        start_time = time.time()
        for _ in range(repeat):
            res = fn(query)
        elapsed_time = time.time() - start_time
        results.setdefault(f"{method}_time", []).append(elapsed_time / repeat)
        recall = get_recall(res, gt)
        results.setdefault(f"{method}_recall", []).append(recall)

    results.setdefault("widths", []).append(width)

results = pd.DataFrame(results)

# %%
# results.to_csv("range_search_fifa.csv", index=False)
results = pd.read_csv("assets/range_search_fifa.csv")
results["frac"] = round_to_nearest_power_fraction(results["widths"])

# %%
# Group by frac and compute mean times
df = results.copy()
grouped = df.groupby("frac").mean(numeric_only=True)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(
    grouped.index,
    grouped["hierarchical_time"],
    marker="x",
    label="Hierarchical Index",
    color="blue",
)
# plt.plot(grouped.index, grouped["rtree_time"], marker="o", label="R-Tree", color='orange')
plt.plot(
    grouped.index,
    grouped["linear_time"],
    marker="s",
    label="Linear Time",
    color="green",
)
plt.plot(
    grouped.index,
    grouped["partition_tree_time"],
    marker="s",
    label="Partition Tree",
    color="purple",
)
# plt.plot(grouped.index, grouped["kdtree_time"], marker="^", label="KD-Tree", color='red')

# Formatting
plt.xlabel("Stripe Range Width (fraction)", fontsize=18)
plt.ylabel("Time (s)", fontsize=18)
plt.title(f"Query Time (FIFA Dataset)", fontsize=22)
plt.legend(fontsize=19)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig(f'assets/range_search_us_fifa.png')
plt.show()
