# %%
import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from time import time

from ranges.stripe_range import StripeRange
from algorithms.search import *

# %% [markdown]
# - Example
# - Experiments

# %% [markdown]
# ## Example

# %%
# Config
n = 1000
dim = 2

# %%
# np.random.seed(0)
points = np.random.rand(n, dim) * 1000

# %% [markdown]
# Generate range

# %%
def generate_stripe_range(points, dim, k):
    """
    Generate a StripeRange with exactly k points inside, choosing start and end randomly.

    Args:
        points (np.ndarray): The set of points.
        dim (int): The dimensionality of the points.
        k (int): The exact number of points to include in the stripe.

    Returns:
        StripeRange: The generated stripe range.
    """
    # Generate a random normal vector
    normal_vector = np.random.rand(dim)

    # Project the points onto the normal vector
    dot_products = np.dot(points, normal_vector)

    # Sort the dot products
    sorted_dots = np.sort(dot_products)

    # Randomly choose a starting index for the range
    max_start_index = len(sorted_dots) - k
    start_index = np.random.randint(0, max_start_index + 1)

    # Select the start and end based on the chosen range
    start_dot = sorted_dots[start_index]
    end_dot = sorted_dots[start_index + k - 1]

    # Calculate start and end points for the stripe
    start = start_dot
    end = end_dot

    # Create the StripeRange
    stripe_range = StripeRange(normal_vector, start, end)

    return stripe_range

# %%
def get_output_size(points, stripe_range):
    # Count the number of points inside the stripe
    inside = sum(1 for point in points if stripe_range.is_in(point))
    return inside

# %%
stripe_range = generate_stripe_range(points, dim, 40)

inside = 0
for point in points:
    if stripe_range.is_in(point):
        inside += 1
print(f"Number of points inside the stripe: {inside}/{n}")

# %%
(graph, epsnet) = preprocess(points, epsnet_size=100, graph_type="random", degree=100)
# (graph, epsnet) = preprocess(points, epsnet_size=100, graph_type="theta", num_directions=8)
# (graph, epsnet) = preprocess(points, epsnet_size=100, graph_type="knn", k=10)
result = query((graph, epsnet), stripe_range)

# %%
epsnet_in_range = []
for ind in epsnet.epsnet_indices:
    if stripe_range.is_in(graph.points[ind]):
        epsnet_in_range.append(graph.points[ind])
len(result), len(epsnet_in_range)

# %% [markdown]
# ## Run Experiments

# %% [markdown]
# ### Random Graph

# %%
n_values = [1000, 2000, 4000, 8000, 16000]
dim_values = [4, 8, 16, 32]
degree_values = [16, 32, 64]
epsnet_size_ratios = [1 / 16, 1 / 128]
output_sizes = [1 / 8, 1 / 128]

results = []

for n in n_values:
    for dim in dim_values:
        for degree in degree_values:
            for epsnet_size_ratio in epsnet_size_ratios:
                for output_size in output_sizes:
                    print(
                        f"Running with n={n}, dim={dim}, degree={degree}, epsnet_size_ratio={epsnet_size_ratio}, output_size={output_size}"
                    )
                    points = np.random.rand(n, dim) * 1000

                    print("Generating stripe range...")
                    stripe = generate_stripe_range(points, dim, k=int(n * output_size))

                    print(f"Preprocessing [random]...")
                    (random_graph, random_epsnet) = preprocess(
                        points,
                        epsnet_size=int(n * epsnet_size_ratio),
                        graph_type="random",
                        degree=degree,
                    )
                    # print(f"Preprocessing [theta]...")
                    # (theta_graph, theta_epsnet) = preprocess(
                    #     points,
                    #     epsnet_size=int(n * epsnet_size_ratio),
                    #     graph_type="theta",
                    #     num_directions=degree,
                    # )
                    print(f"Preprocessing [knn]...")
                    (knn_graph, knn_epsnet) = preprocess(
                        points,
                        epsnet_size=int(n * epsnet_size_ratio),
                        graph_type="knn",
                        k=degree,
                    )

                    print("Random graph...")
                    start = time()
                    random_result = query((random_graph, random_epsnet), stripe)
                    end = time()
                    random_time = end - start

                    # print("Theta graph...")
                    # start = time()
                    # theta_result = query((theta_graph, theta_epsnet), stripe)
                    # end = time()
                    # theta_time = end - start

                    print("KNN graph...")
                    start = time()
                    knn_result = query((knn_graph, knn_epsnet), stripe)
                    end = time()
                    knn_time = end - start

                    print("linear search...")
                    start = time()
                    size = get_output_size(points, stripe)
                    linear_result = [i for i in range(size)]
                    end = time()
                    linear_time = end - start

                    random_recall = len(random_result) / size
                    # theta_recall = len(theta_result) / size
                    knn_recall = len(knn_result) / size
                    linear_recall = len(linear_result) / size
                    print(
                        f"recalls: random {random_recall}, knn {knn_recall}, linear {linear_recall}"
                    )

                    results.append(
                        {
                            "n": n,
                            "dim": dim,
                            "degree": degree,
                            "epsnet_size_ratio": epsnet_size_ratio,
                            "output_size": output_size,
                            "random_recall": random_recall,
                            "theta_recall": None,
                            "knn_recall": knn_recall,
                            "linear_recall": linear_recall,
                            "random_time": random_time,
                            "theta_time": None,
                            "knn_time": knn_time,
                            "linear_time": linear_time,
                        }
                    )

# %%
import pandas as pd
df = pd.DataFrame(results)
df.to_csv("results_no_theta.csv", index=False)


