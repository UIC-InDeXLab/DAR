import numpy as np

from algorithms.epsnet import EpsNet
from graphs.random_graph import RandomGraph
from graphs.theta_graph import ThetaGraph
from graphs.knn_graph import KNNGraph
from algorithms.bfs import bfs


def preprocess(points: np.ndarray, epsnet_size, graph_type, **graph_args):
    """
    graph_type: random or theta
    """
    epsnet = EpsNet(points, epsnet_size=epsnet_size)
    if graph_type == "random":
        graph = RandomGraph(points, **graph_args)  # probability
    elif graph_type == "theta":
        graph = ThetaGraph(points, **graph_args)  # num_directions
    elif graph_type == "knn":
        graph = KNNGraph(points, **graph_args)  # k

    return (graph, epsnet)


def query(data, range):
    graph = data[0]
    epsnet = data[1]
    # find starting nodes
    start_nodes = []
    for index in epsnet.epsnet_indices:
        if range.is_in(graph.points[index]):
            start_nodes.append(index)
    # bfs
    in_range_nodes = bfs(graph, start_nodes, range)
    return in_range_nodes
