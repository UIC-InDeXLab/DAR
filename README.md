#  Efficient Direct-Access Ranked Retrieval
[The Technical Report](/technical_report.pdf)
## Structure
* `methods/kth` contains the implementation of the following algorithms:
    * `EpsRange`
    * `EpsHier`
    * `KthLevel`
    * Baselines `TA` and `Fagin`
* `methods/range_search` contains the implementation of the following algorithms:
    * `Hierarchical Sampling`
    * Baseline `Partition Tree`
    * Baseline `KD-tree`
    * Baseline `R-tree`
* `experiments` contains example codes for using these algorithms.
