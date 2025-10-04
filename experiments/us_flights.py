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


import kagglehub

dataset = kagglehub.dataset_download("usdot/flight-delays")

# %%
df = pd.read_csv(f"{dataset}/flights.csv")
df.shape

# %%
df.head()

# %%
df.columns

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

# Step 1: Drop unwanted columns
drop_cols = [
    "YEAR",
    "FLIGHT_NUMBER",
    "TAIL_NUMBER",
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
results.to_csv("kth_flight_delays.csv", index=False)
