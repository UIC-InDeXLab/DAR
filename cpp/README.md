This directory contains the baseline implementations (whatever already exists) as the state-of-the-art for answering 'Stripe Range Searching'.

- CGAL:
    - Partition Tree
    - Halfspace intersection
- Python:
    - KDTree with bounding box
    - RTree with bounding box
    - First find MBR covers then apply RTree
        - We should implement this ourselves, I haven't found any references for this yet.
- Brute force

## TODO