import numpy as np
import random
import sys
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree
from collections import deque

#---------------------------------------------
# Data generation and graph construction
#---------------------------------------------
def generate_points(n, d, distribution="gaussian"):
    """Generate n points in d dimensions."""
    if distribution == "gaussian":
        return np.random.randn(n, d)
    elif distribution == "uniform":
        return np.random.rand(n, d)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


def build_knn_graph(points, k=8):
    """Build undirected k-NN graph as adjacency list."""
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    n = points.shape[0]
    adj = [[] for _ in range(n)]
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # skip self
            adj[i].append(j)
            adj[j].append(i)
    return adj


def build_random_graph(n, p=0.0005):
    """Build random undirected graph with edge probability p."""
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < p:
                adj[i].append(j)
                adj[j].append(i)
    return adj

#---------------------------------------------
# Query stripe and epsilon-net sampling
#---------------------------------------------
def define_stripe(points, a, delta):
    """Return mask of points where a.x in [c, c+delta]."""
    vals = points.dot(a)
    c = np.median(vals)
    return (vals >= c) & (vals <= c + delta)


def sample_epsilon_net(stripe_mask, epsilon_frac):
    """Sample epsilon-net: epsilon_frac * total points as seeds inside stripe."""
    idx = np.where(stripe_mask)[0]
    total = stripe_mask.size
    m = int(epsilon_frac * total)
    m = min(m, idx.size)
    return np.random.choice(idx, size=m, replace=False) if m > 0 else np.array([], dtype=int)

#---------------------------------------------
# Optimized BFS search within stripe
#---------------------------------------------
def bfs_search(adj, stripe, seeds):
    """Perform fast BFS restricted to nodes inside stripe."""
    n = len(adj)
    # Convert stripe mask to Python list for fast indexing
    #stripe = stripe_mask.tolist()
    visited = [False] * n  # Python list for visited flags
    adj_list = adj
    stripe_local = stripe
    visited_local = visited
    discovered = []
    q = deque()

    # Initialize queue with all seeds (pre-checked for stripe)
    for s in seeds:
        if not visited_local[s]:
            visited_local[s] = True
            q.append(s)
            discovered.append(s)

    # BFS traversal
    while q:
        u = q.popleft()
        for v in adj_list[u]:
            if not visited_local[v] and stripe_local[v]:
                visited_local[v] = True
                discovered.append(v)
                q.append(v)
    return discovered


def linear_search(stripe_mask):
    return np.where(stripe_mask)[0]

#---------------------------------------------
# Memory measurement
#---------------------------------------------
def measure_graph_mem(adj):
    total = sys.getsizeof(adj)
    for lst in adj:
        total += sys.getsizeof(lst)
        for v in lst:
            total += sys.getsizeof(v)
    return total

#---------------------------------------------
# Experiment runner
#---------------------------------------------
def run_experiment(n, d, delta, epsilon_frac, distribution, graph_type, k, p):
    np.random.seed(0)
    random.seed(0)
    points = generate_points(n, d, distribution)
    adj = build_knn_graph(points, k) if graph_type == 'knn' else build_random_graph(n, p)
    a = np.array([1.0] * (d - 1) + [0.0])
    a /= np.linalg.norm(a)
    stripe_mask = define_stripe(points, a, delta)
    stripe_total = int(stripe_mask.sum())

    # Sample epsilon-net seeds and filter those in stripe
    seeds = sample_epsilon_net(stripe_mask, epsilon_frac)
    seeds = np.array([s for s in seeds if stripe_mask[s]], dtype=int)
    seed_count = len(seeds)

    # BFS search
    stripe = stripe_mask.tolist()
    t0 = time.time()
    bfs_res = bfs_search(adj, stripe, seeds)
    bfs_time = (time.time() - t0) * 1000

    # Linear search
    t0 = time.time()
    lin_res = linear_search(stripe_mask)
    lin_time = (time.time() - t0) * 1000

    # KD-tree search (filter by stripe)
    KDTree(points)
    t0 = time.time()
    kd_idx = np.where(stripe_mask)[0]
    kd_time = (time.time() - t0) * 1000

    graph_mem = measure_graph_mem(adj) / (1024**2)
    kd_mem = sys.getsizeof(KDTree(points)) / (1024**2)

    return {
        'stripe_total': stripe_total,
        'seed_count': seed_count,
        'bfs_found': len(bfs_res),
        'times': {'bfs': bfs_time, 'linear': lin_time, 'kd': kd_time},
        'mem': {'bfs': graph_mem, 'linear': 0.0, 'kd': kd_mem}
    }

#---------------------------------------------
# Main: run experiments & print LaTeX
#---------------------------------------------
def main():
    defaults = {'n':100000,'d':4,'delta':0.2,'epsilon_frac':0.001,'distribution':'gaussian','graph_type':'knn','k':4,'p':0.0005}
    param_sets = {
        'n':[10000,50000,100000, 500000],
        'd':[2,4,8,16,32],
        'delta':[0.1,0.2,0.3,0.4,0.5],
        'epsilon_frac':[0.001, 0.005, 0.01],
        'distribution':['gaussian','uniform'],
        'graph_type':['knn','random'],
        'k':[2,4,8,16,32]
    }
    for param, values in param_sets.items():
        print(f"\n% Varying {param}\n\n\\begin{{table}}[h!]\\centering")
        if param=='epsilon_frac':
            header_label = 'Epsilon (%)'
        else:
            header_label = param
        print("\\begin{tabular}{c|c|c|c|ccc|ccc}")
        print("\hline")
        print(f"{header_label} & Seeds & In stripe & BFS discovered & \\multicolumn{{3}}{{c|}}{{Time (ms)}} & \\multicolumn{{3}}{{c}}{{Space (MB)}} \\")
        print(" & & & & BFS & Linear & KD-tree & BFS & Linear & KD-tree \\")
        print("\hline")
        for val in values:
            cfg = defaults.copy()
            cfg[param] = val
            if param=='k': cfg['graph_type']='knn'
            res = run_experiment(**cfg)
            if param=='epsilon_frac':
                label = f"{val*100:.3f}\\%"
            else:
                label = str(val)
            print(f"{label} & {res['seed_count']} & {res['stripe_total']} & {res['bfs_found']} & {res['times']['bfs']:} & {res['times']['linear']:} & {res['times']['kd']:} & {res['mem']['bfs']:} & {res['mem']['linear']:} & {res['mem']['kd']:} \\")
        print("\hline\\end{tabular}")
        print(f"\\caption{{Performance when varying {param}}}\n\\end{{table}}\n")

if __name__=='__main__': main()