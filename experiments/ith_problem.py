# %%
import sys

sys.path.append("..")

import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np
import time

from methods.kth.epssample import *
from methods.kth.fagin import *
from methods.kth.threshold import *
from methods.utils import *
from methods.range_search.hierarchy import *
from methods.range_search.partition_tree import PartitionTree, Point
from ranges.stripe_range import StripeRange
from methods.kth import epssample
from methods.kth.klevel import *

# %% [markdown]
# Utils


# %%
def hier_index(points):
    ind = HierarchicalIndex(points, decay=4)
    ind.build_index()
    ind.find_coverage()
    ind.find_neighbor_stats()
    return ind


def partition_tree_index(points):
    return PartitionTree(Point.from_numpy(points))


def build_methods(points, eps):
    n = len(points)

    print("Building hierarchical...")
    eps_hier_prep = preprocess(points, hier_index, eps)
    print("Building partition tree...")
    eps_range_prep = preprocess(points, partition_tree_index, eps)

    partition_tree_query_fn = lambda index, stripe: index.halfspace_query(
        stripe.normal_vector.tolist(), stripe.start_dot, stripe.end_dot
    )

    # TODO: exhaustive should always be the first method
    return {
        "Exhaustive": lambda weights, k, eps, epssample: find_kth_exhaustive(
            points, weights, k
        ),
        "EpsHier": lambda weights, k, eps, epssample: find_kth(
            eps_hier_prep[0],
            # eps_hier_prep[1],
            epssample,
            eps,
            n,
            weights,
            k,
            query_fn=lambda index, stripe: index.query(stripe),
        ),
        "EpsRange": lambda weights, k, eps, epssample: find_kth(
            eps_range_prep[0],
            # eps_range_prep[1],
            epssample,
            eps,
            n,
            weights,
            k,
            query_fn=partition_tree_query_fn,
        ),
        "Fagin": lambda weights, k, eps, epssample: fagins_algorithm(
            points, weights, k
        ),
        "TA": lambda weights, k, eps, epssample: threshold_algorithm(
            points, weights, k
        ),
    }


# %% [markdown]
# ### Time vs k and dim:

# %%
ns = [10_000, 100_000, 200_000, 500_000, 1_000_000]
# dims = [2, 4, 8, 32, 128]
dims = [16]
# ks = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]
ks = [1 / 8, 1 / 32, 1 / 64]

epss = [1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512]
repeats = 5

results = {}

for n in ns:
    for dim in tqdm(dims, desc=f"Running for n={n} / {ns}"):
        points = np.random.zipf(1.5, size=(n, dim))
        weights = np.random.rand(dim)
        weights /= np.linalg.norm(weights)

        fns = build_methods(points, epss[0])  # Build methods with the first eps

        for eps in epss:
            tqdm.write(f"Building index for eps: {eps} / {epss} ...")
            # fns = build_methods(points, eps)

            sample = epssample.epsilon_sample(points, epsilon=eps)

            for l in ks:
                k = int(l * n)
                tqdm.write(f"Running for k={k} / {ks}...")

                gt = None

                for method, fn in fns.items():
                    tqdm.write(f"Running {method}...")
                    start = time.time()
                    for _ in range(repeats):
                        res = fn(weights, k, eps, sample)
                    end = time.time()

                    if method == "Exhaustive":
                        gt = res

                    results.setdefault(f"{method}_time", []).append(
                        (end - start) / repeats
                    )

                    if method in ["EpsRange", "EpsHier"]:
                        try:
                            if gt[0] in points[res]:
                                results.setdefault(f"{method}_correct", []).append(1)
                                tqdm.write("correct")
                            else:
                                results.setdefault(f"{method}_correct", []).append(0)
                                tqdm.write("incorrect")
                        except Exception as e:
                            results.setdefault(f"{method}_correct", []).append(0)
                            tqdm.write(f"Error: {e}")

                results.setdefault("eps", []).append(eps)

                results.setdefault("k", []).append(k)
                results.setdefault("n", []).append(n)
                results.setdefault("dim", []).append(dim)

results = pd.DataFrame(results)

# %% [markdown]
# Visualize

# %%
df = pd.read_csv("kth_query_vs_dims.csv")
df.head()

# %%
# Group by dim and compute mean time
grouped = df.groupby("dim").mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(
    grouped["dim"],
    grouped["EpsHier_time"],
    marker="o",
    label="EpsHier",
    color="blue",
)
plt.plot(
    grouped["dim"],
    grouped["EpsRange_time"],
    marker="o",
    label="EpsRange",
    color="orange",
)
plt.plot(
    grouped["dim"],
    grouped["Exhaustive_time"],
    marker="s",
    label="Exhaustive Search",
    color="green",
)
plt.plot(
    grouped["dim"], grouped["Fagin_time"], marker="^", label="Fagin", color="purple"
)
plt.plot(grouped["dim"], grouped["TA_time"], marker="x", label="TA", color="brown")

plt.xlabel("Dimension", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title("Query Times vs Dimension ($n = 10000$)", fontsize=22)
plt.legend(fontsize=17)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig("assets/kth_element_query_dims.png")
plt.show()

# %% [markdown]
# ### Visualization: Time vs. K

# %%
# Group by k and compute mean time
grouped = df[df["dim"] == 32].groupby("k").mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(
    grouped["k"],
    grouped["EpsHier_time"],
    marker="o",
    label="EpsHier",
    color="blue",
)
plt.plot(
    grouped["k"],
    grouped["EpsRange_time"],
    marker="o",
    label="EpsRange",
    color="orange",
)
plt.plot(
    grouped["k"],
    grouped["Exhaustive_time"],
    marker="s",
    label="Exhaustive Search",
    color="green",
)
plt.plot(grouped["k"], grouped["Fagin_time"], marker="^", label="Fagin", color="purple")
plt.plot(grouped["k"], grouped["TA_time"], marker="x", label="TA", color="brown")

plt.xlabel("i (ith point retrieval)", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title("Query Times vs i ($n = 10000, d = 32$)", fontsize=22)
plt.legend(fontsize=17, loc="upper left")
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig("assets/kth_element_query_k.png")
plt.show()

# %% [markdown]
# ### Visualize time vs. size:

# %%
# df = pd.concat([
# pd.read_csv("kth_results_sizes_128_odd.csv"),
# pd.read_csv("kth_results_sizes_128_even.csv"),
# ])
df = pd.read_csv("assets/kth_time_size.csv")
df.head()

# %%
# Group by k and compute mean time
grouped = df.groupby("eps").mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(
    grouped["eps"],
    grouped["EpsHier_time"],
    marker="o",
    label="EpsHier",
    color="blue",
)
plt.plot(
    grouped["eps"],
    grouped["EpsRange_time"],
    marker="o",
    label="EpsRange",
    color="orange",
)
plt.plot(
    grouped["eps"],
    grouped["Exhaustive_time"],
    marker="s",
    label="Exhaustive",
    color="green",
)
plt.plot(
    grouped["eps"], grouped["Fagin_time"], marker="^", label="Fagin", color="purple"
)
plt.plot(grouped["eps"], grouped["TA_time"], marker="x", label="TA", color="brown")

plt.xlabel(r"Value of $\varepsilon$", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title(r"Query Time vs $\varepsilon$", fontsize=22)
# plt.legend(fontsize=17)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig("assets/kth_time_vs_eps.png")
plt.show()

# %% [markdown]
# ### Real dataset US Used Cars:

# %%
import kagglehub

dataset = kagglehub.dataset_download("ananaymital/us-used-cars-dataset")

# %%
df = pd.read_csv(f"{dataset}/used_cars_data.csv")
df.head()

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
ks = [1 / 8, 1 / 32, 1 / 64]
epss = [1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512]
repeats = 5

results = {}

n = data.shape[0]
dim = data.shape[1]
points = data
weights = np.random.rand(dim)
weights /= np.linalg.norm(weights)

fns = build_methods(points, epss[0])  # Build methods with the first eps

for eps in epss:
    tqdm.write(f"Building index for eps: {eps} / {epss} ...")

    sample = epssample.epsilon_sample(points, epsilon=eps)

    for l in ks:
        k = int(l * n)
        tqdm.write(f"Running for k={k} / {ks}...")

        gt = None

        for method, fn in fns.items():
            if method == "Fagin":
                continue
            tqdm.write(f"Running {method}...")
            start = time.time()
            for _ in range(repeats):
                res = fn(weights, k, eps, sample)
            end = time.time()

            if method == "Exhaustive":
                gt = res

            results.setdefault(f"{method}_time", []).append((end - start) / repeats)

            if method in ["EpsRange", "EpsHier"]:
                try:
                    if gt[0] in points[res]:
                        results.setdefault(f"{method}_correct", []).append(1)
                        tqdm.write("correct")
                    else:
                        results.setdefault(f"{method}_correct", []).append(0)
                        tqdm.write("incorrect")
                except Exception as e:
                    results.setdefault(f"{method}_correct", []).append(0)
                    tqdm.write(f"Error: {e}")

        results.setdefault("eps", []).append(eps)

        results.setdefault("k", []).append(k)

results = pd.DataFrame(results)

# %% [markdown]
# Visualize

# %%
results = pd.read_csv("assets/kth_used_cars.csv")

# %%
# Copy the dataframe (optional if df is already your desired data)
df = results.copy()
df = df.groupby("eps").mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(
    df["eps"],
    df["EpsHier_time"],
    marker="o",
    label="EpsHier",
    color="blue",
)
plt.plot(
    df["eps"],
    df["EpsRange_time"],
    marker="o",
    label="EpsRange",
    color="orange",
)
plt.plot(
    df["eps"],
    df["Exhaustive_time"],
    marker="s",
    label="Exhaustive Search",
    color="green",
)
plt.plot(
    df["eps"],
    df["TA_time"],
    marker="x",
    label="TA",
    color="brown",
)

plt.xlabel(r"$\varepsilon$", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title(r"Query Times vs $\varepsilon$ (US Used Cars dataset)", fontsize=22)
plt.legend(fontsize=17, loc="upper left")
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig("assets/kth_US_used_cars_time_eps.png")
plt.show()


# %%
# Copy the dataframe (optional if df is already your desired data)
df = results.copy()
df = df.groupby("k").mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(
    df["k"],
    df["EpsHier_time"],
    marker="o",
    label="EpsHier",
    color="blue",
)
plt.plot(
    df["k"],
    df["EpsRange_time"],
    marker="o",
    label="EpsRange",
    color="orange",
)
plt.plot(
    df["k"],
    df["Exhaustive_time"],
    marker="s",
    label="Exhaustive Search",
    color="green",
)
plt.plot(
    df["k"],
    df["TA_time"],
    marker="x",
    label="TA",
    color="brown",
)

plt.xlabel("i (ith point retrieval)", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title(r"Query Times vs $i$ (US Used Cars dataset)", fontsize=22)
# plt.legend(fontsize=17)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig("assets/kth_US_used_cars_time_k.png")
plt.show()


# %% [markdown]
# ### Real dataset: FIFA

# %%
import kagglehub

# Download latest version
path = kagglehub.dataset_download("stefanoleone992/fifa-23-complete-player-dataset")

# %%
df = pd.read_csv(f"{path}/male_teams.csv")
df.shape

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
ks = [1 / 8, 1 / 32, 1 / 64]
epss = [1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256, 1 / 512]
repeats = 5

results = {}

n = data.shape[0]
dim = data.shape[1]
points = data
weights = np.random.rand(dim)
weights /= np.linalg.norm(weights)

fns = build_methods(points, epss[0])  # Build methods with the first eps

for eps in epss:
    tqdm.write(f"Building index for eps: {eps} / {epss} ...")

    sample = epssample.epsilon_sample(points, epsilon=eps)

    for l in ks:
        k = int(l * n)
        tqdm.write(f"Running for k={k} / {ks}...")

        gt = None

        for method, fn in fns.items():
            if method == "Fagin":
                continue
            tqdm.write(f"Running {method}...")
            start = time.time()
            for _ in range(repeats):
                res = fn(weights, k, eps, sample)
            end = time.time()

            if method == "Exhaustive":
                gt = res

            results.setdefault(f"{method}_time", []).append((end - start) / repeats)

            if method in ["EpsRange", "EpsHier"]:
                try:
                    if gt[0] in points[res]:
                        results.setdefault(f"{method}_correct", []).append(1)
                        tqdm.write("correct")
                    else:
                        results.setdefault(f"{method}_correct", []).append(0)
                        tqdm.write("incorrect")
                except Exception as e:
                    results.setdefault(f"{method}_correct", []).append(0)
                    tqdm.write(f"Error: {e}")

        results.setdefault("eps", []).append(eps)

        results.setdefault("k", []).append(k)

results = pd.DataFrame(results)

# %%
results.to_csv("kth_fifa.csv", index=False)

# %%
results.head()

# %%
# Copy the dataframe (optional if df is already your desired data)
df = results.copy()
df = df.groupby("eps").mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(
    df["eps"],
    df["EpsHier_time"],
    marker="o",
    label="EpsHier",
    color="blue",
)
plt.plot(
    df["eps"],
    df["EpsRange_time"],
    marker="o",
    label="EpsRange",
    color="orange",
)
plt.plot(
    df["eps"],
    df["Exhaustive_time"],
    marker="s",
    label="Exhaustive Search",
    color="green",
)
plt.plot(
    df["eps"],
    df["TA_time"],
    marker="x",
    label="TA",
    color="brown",
)

plt.xlabel(r"$\varepsilon$", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title(r"Query Times vs $\varepsilon$ (FIFA dataset)", fontsize=22)
plt.legend(fontsize=17)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig("assets/kth_FIFA_time_eps.png")
plt.show()


# %%
# Copy the dataframe (optional if df is already your desired data)
df = results.copy()
df = df.groupby("k").mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(
    df["k"],
    df["EpsHier_time"],
    marker="o",
    label="EpsHier",
    color="blue",
)
plt.plot(
    df["k"],
    df["EpsRange_time"],
    marker="o",
    label="EpsRange",
    color="orange",
)
plt.plot(
    df["k"],
    df["Exhaustive_time"],
    marker="s",
    label="Exhaustive Search",
    color="green",
)
plt.plot(
    df["k"],
    df["TA_time"],
    marker="x",
    label="TA",
    color="brown",
)

plt.xlabel("i (ith point retrieval)", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title(r"Query Times vs $i$ (FIFA dataset)", fontsize=22)
plt.legend(fontsize=17)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig("assets/kth_FIFA_time_k.png")
plt.show()


# %% [markdown]
# ### Kth Level of Arrangement:


# %%
def build_klevel(points):
    n = len(points)
    dual_lines = []
    for p in points:
        a, b = p
        dual_lines.append((a, b, -1))

    kthlevels = {}
    for k in tqdm(range(1, n // 2)):
        kthlevels[k] = get_kth_level_arrangement(dual_lines, k)

    return kthlevels


# %%
ns = [100, 200, 400, 800, 1000, 2000, 4000]
dim = 2
ks = [1 / 4, 1 / 8, 1 / 16, 1 / 32]
epss = [1 / 16]

repeats = 5

results = {}

for n in ns:
    points = np.random.zipf(1.5, size=(n, dim))
    weights = np.random.rand(dim)
    weights /= np.linalg.norm(weights)

    fns = build_methods(points, epss[0])  # Build methods with the first eps

    print("Building kthlevel...")
    start = time.time()
    kthlevels = build_klevel(points)
    end = time.time()
    build_time = end - start

    for eps in epss:
        tqdm.write(f"Building index for eps: {eps} / {epss} ...")

        sample = epssample.epsilon_sample(points, epsilon=eps)

        for l in ks:
            k = int(l * n)
            tqdm.write(f"Running for k={k} / {ks}...")

            gt = None

            # kth level method
            tqdm.write("Running KthLevel...")
            start = time.time()
            for _ in range(repeats):
                angle = np.arctan2(weights[1], weights[0])
                find_intersecting_segment_optimized(kthlevels[k], angle)
            end = time.time()
            results.setdefault("KthLevel_time", []).append((end - start) / repeats)
            results.setdefault("KthLevel_size", []).append(
                get_size_recursive(kthlevels[k])
            )
            results.setdefault("KthLevel_build_time", []).append(build_time)

            for method, fn in fns.items():
                tqdm.write(f"Running {method}...")
                start = time.time()
                for _ in range(repeats):
                    res = fn(weights, k, eps, sample)
                end = time.time()

                if method == "Exhaustive":
                    gt = res
                    results.setdefault(f"{method}_size", []).append(
                        get_size_recursive(points)
                    )

                results.setdefault(f"{method}_time", []).append((end - start) / repeats)

                if method in ["EpsRange", "EpsHier"]:
                    try:
                        if gt[0] in points[res]:
                            results.setdefault(f"{method}_correct", []).append(1)
                            tqdm.write("correct")
                        else:
                            results.setdefault(f"{method}_correct", []).append(0)
                            tqdm.write("incorrect")
                    except Exception as e:
                        results.setdefault(f"{method}_correct", []).append(0)
                        tqdm.write(f"Error: {e}")

            results.setdefault("eps", []).append(eps)

            results.setdefault("k", []).append(k)
            results.setdefault("n", []).append(n)
            results.setdefault("dim", []).append(dim)

results = pd.DataFrame(results)

# %%
# results.to_csv("kth_klevel.csv", index=False)
results = pd.read_csv("kth_klevel_prep_sizes.csv")

# %%
results.head()

# %%
# Group by dim and compute mean time
df = results.copy()
grouped = df.groupby("n").mean().reset_index()

# Plotting
plt.figure(figsize=(8, 6))
# plt.plot(
#     grouped["n"],
#     grouped["EpsHier_time"],
#     marker="o",
#     label="EpsHier",
#     color="blue",
# )
# plt.plot(
#     grouped["n"],
#     grouped["EpsRange_time"],
#     marker="o",
#     label="EpsRange",
#     color="orange",
# )
# plt.plot(
#     grouped["n"],
#     grouped["Exhaustive_time"],
#     marker="s",
#     label="Exhaustive Search",
#     color="green",
# )
# plt.plot(
#     grouped["n"],
#     grouped["KthLevel_time"],
#     marker="s",
#     label="KthLevel",
#     color="red",
# )
# plt.plot(
#     grouped["n"], grouped["Fagin_time"], marker="^", label="Fagin", color="purple"
# )
# plt.plot(grouped["n"], grouped["TA_time"], marker="x", label="TA", color="brown")
plt.plot(
    grouped["n"],
    grouped["KthLevel_BuildTime"],
    marker="o",
    label="KthLevel",
    color="red",
)
plt.plot(
    grouped["n"],
    grouped["KthLevel_HierTime"],
    marker="o",
    label="EpsHier",
    color="blue",
)
plt.plot(
    grouped["n"],
    grouped["KthLevel_RangeTime"],
    marker="o",
    label="EpsRange",
    color="orange",
)
plt.xlabel(r"Dataset Size $n$", fontsize=18)
plt.ylabel("Time", fontsize=18)
plt.title("Preprocess Time vs Dataset Size ($d = 2$)", fontsize=22)
plt.legend(fontsize=17)
plt.grid(True)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xticks(fontsize=16)  # You can adjust the number as needed
plt.yticks(fontsize=16)
plt.tight_layout()
# plt.savefig("assets/kthlevel_prep_time.png")
plt.show()
