# Random-Access Ranked Retrieval and Similarity Search

This repository contains implementations and experimental code for efficient Random-Access Ranked Retrieval (RAR) and Random-Access Similarity Search (RAS) algorithms.

## Installation

Requires Python 3.9+.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the project as an editable package so `ranges`, `methods`,
#    and `knn` import from anywhere (recommended)
pip install -e .
```

Optional extras:

```bash
pip install -e ".[profiling]"  # psutil, pympler, miniball — memory-usage experiments
pip install -e ".[test]"       # pytest — test suite
```

## Repository Structure

### Core Implementations

#### `methods/kth/`
Contains implementations of RAR algorithms:
- **`EpsRange`** - Epsilon-based range approach
- **`EpsHier`** - Epsilon-based hierarchical method
- **`KthLevel`** - Level-based k-th element algorithm
- **Baselines:** `TA` (Threshold Algorithm) and `Fagin`

#### `methods/range_search/`
Contains implementations of range search algorithms:
- **`Hierarchical Sampling`** - Novel hierarchical sampling approach
- **Baseline algorithms:** `Partition Tree`, `KD-tree`, `R-tree`

#### `knn/methods/`
Contains the implementations of the RAS problem on Euclidean distance and cosine similarity.
- **`EpsHier`** - Our algorithm for random-access similarity search
- **`Faiss`** - Common ANN indexes as baselines for random-access similarity search

### Experimental Code

#### `experiments/`
Code to reproduce results for Random-Access Ranked Retrieval experiments.

#### `knn/experiments/`
Code to reproduce results for Random-Access Similarity Search experiments.

## Datasets

RAS experiments use standard [ANN-Benchmarks](http://ann-benchmarks.com) datasets
(e.g. `glove-100-angular`, `sift-128-euclidean`, `fashion-mnist-784-euclidean`).
They are downloaded automatically on first use into `knn/datasets/data/`
(git-ignored) by `knn/datasets/data_loader.py`. Some RAR experiments use the
US Flights dataset, which is fetched via `kagglehub`.

## Running Experiments

The experiment files use [`# %%`](https://code.visualstudio.com/docs/python/jupyter-support-py)
cell markers, so they can be run interactively (VS Code Jupyter / Jupyter) or
as plain scripts. Run them from the repository root, for example:

```bash
# Random-Access Ranked Retrieval
python experiments/ith_problem.py        # k-th element problem
python experiments/srs_problem.py        # simple range search
python experiments/us_flights.py         # US Flights dataset
python experiments/memory_usage.py       # memory profiling (needs [profiling] extra)

# Random-Access Similarity Search
python knn/experiments/time_recall_benchmark.py   # time vs. recall
python knn/experiments/space_time_benchmark.py    # space vs. time
```

## Tests

```bash
pytest knn/tests
```

## Citation

If you use this code, please cite our paper (see [`CITATION.cff`](CITATION.cff)).

## License

Released under the [MIT License](LICENSE).
