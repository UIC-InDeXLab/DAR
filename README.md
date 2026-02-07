# Random-Access Ranked Retrieval and Similarity Search

This repository contains implementations and experimental code for efficient Random-Access Ranked Retrieval (RAR) and Random-Access Similarity Search algorithms (RAS).

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

#### `knn/methods`
Contains the implementations of RAS problem on Euclidean distance and cosine similarity.
- **`EpsHier`** - Our algorithm on random-access similarity search
- **`Faiss`** - Common ANN indexes as baselines for random-access similarity search

### Experimental Code

#### `experiments/`
Contains code to reproduce results for Random-access Ranked Retrieval experiments.

#### `knn/experiments`
Contains code to reproduce results for Random-access Similarity search experiments.

### Cite