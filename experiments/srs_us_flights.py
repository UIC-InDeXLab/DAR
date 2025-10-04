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
from methods.range_search.ball_tree import *
from methods.utils import linear_search
from ranges.stripe_range import StripeRange


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

    # Ball Tree
    print("Indexing Ball Tree...")
    ball_tree = BallTreeIndex(points=points, leaf_size=40)
    ball_tree.build_index()

    # KDTree
    print("Indexing KDTree...")
    kdtree = KDTree(points=points)

    # RTree
    print("Indexing RTree...")
    rtree = RTree(dimensions=dim)
    rtree.insert_points(points)

    # Partition Tree
    print("Indexing Partition Tree...")
    partition_tree = PartitionTree(Point.from_numpy(points))

    return {
        "hierarchical": lambda q: hie.query(q),
        "ball_tree": lambda q: ball_tree.query(q),
        "kdtree": lambda q: kdtree.query(q),
        "rtree": lambda q: rtree.query(q),
        "partition_tree": lambda q: partition_tree.halfspace_query(
            q.normal_vector.tolist(), q.start_dot, q.end_dot
        ),
    }, points


def sample_query(points, width):
    return StripeRange.sample_stripe(points, r=width, tolerance=0.001)


def get_recall(results, gt):
    return len(results) / len(gt) if len(gt) > 0 else 1.0


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
results.to_csv("srs_flight_delays.csv", index=False)