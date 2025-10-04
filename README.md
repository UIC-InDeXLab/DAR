# Efficient Direct-Access Ranked Retrieval

This repository contains implementations and experimental code for efficient direct-access ranked retrieval algorithms. For detailed technical information, see the [Technical Report](/technical_report.pdf).

## Repository Structure

### Core Implementations

#### `methods/kth/`
Contains implementations of k-th element retrieval algorithms:
- **`EpsRange`** - Epsilon-based range approach
- **`EpsHier`** - Epsilon-based hierarchical method  
- **`KthLevel`** - Level-based k-th element algorithm
- **Baselines:** `TA` (Threshold Algorithm) and `Fagin`

#### `methods/range_search/`
Contains implementations of range search algorithms:
- **`Hierarchical Sampling`** - Novel hierarchical sampling approach
- **Baseline algorithms:** `Partition Tree`, `KD-tree`, `R-tree`

#### `methods/utils.py`
Utility functions and helper methods used across different algorithms.

### Experimental Code

#### `experiments/`
Example codes and benchmarks demonstrating algorithm usage:
- Performance comparisons and analysis
- Real-world dataset evaluations
- Memory usage studies
- Visualization tools

### Supporting Files

#### `ranges/`
Range-related data structures and utilities:
- Base range implementations
- Stripe range operations
