from collections import deque
from graphs.base import Graph
from ranges.base import Range


def bfs(graph: Graph, start_nodes, range: Range):
    """
    Run a BFS starting from a set of start nodes and continue until all the neighbors are outside the range.
    """
    visited = set(start_nodes)
    queue = deque(start_nodes)
    in_range_nodes = []

    while queue:
        node = queue.popleft()
        if range.is_in(graph.points[node]):
            in_range_nodes.append(node)

            for neighbor in graph.adj_matrix[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return in_range_nodes
