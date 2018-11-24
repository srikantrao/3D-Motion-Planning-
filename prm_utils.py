import numpy as np
import matplotlib.pyplot as plt
from sampling import Sampler
from shapely.geometry import Polygon, Point, LineString
from queue import PriorityQueue
import numpy.linalg as LA
from sklearn.neighbors import KDTree
import networkx as nx
import time

## Global Variables
NUM_SAMPLES = 500
NUM_NODES = 10

def can_connect(n1, n2, polygons):
    l = LineString([n1, n2])
    for p in polygons:
        if p.crosses(l) and p.height >= min(n1[2], n2[2]):
            return False
    return True

def sample_nodes(data, grid_start, grid_goal, debug):
    start_time = time.time()
    sampler = Sampler(data)
    # Extract all the polygons
    polygons = sampler._polygons
    nodes = sampler.sample(grid_start, grid_goal, NUM_SAMPLES)
    stop_time = time.time()
    if debug:
        print("Time taken to build Sampler is: {0:5.2f}s".format(stop_time - start_time))
    return nodes, polygons

def heuristic(n1, n2):
    """Returns the Euclidean distance between the points n1 and n2."""
    return LA.norm(np.array(n2) - np.array(n1))

def create_graph(nodes, k, polygons, debug):
    start_time = time.time()
    g = nx.Graph()
    tree = KDTree(nodes)
    for n1 in nodes:
        # for each node connect try to connect to k nearest nodes
        idxs = tree.query([n1], k, return_distance=False)[0]

        for idx in idxs:
            n2 = nodes[idx]
            if n2 == n1:
                continue

            if can_connect(n1, n2, polygons):
                g.add_edge(n1, n2, weight=1)
    stop_time = time.time()
    if debug:
        print("Time taken to create the Graph is: {0:5.2f}s".format(stop_time - start_time))
    return g


def visualize_graph(grid, data, path, grid_start, grid_goal):

    plt.imshow(grid, cmap='Greys', origin='lower')

    nmin = np.min(data[:, 0])
    emin = np.min(data[:, 1])
    plt.plot(grid_start[1], grid_start[0], 'x')
    plt.plot(grid_goal[1], grid_goal[0], 'o')

    path_pairs = zip(path[:-1], path[1:])
    for (n1, n2) in path_pairs:
        plt.plot([n1[1] - emin, n2[1] - emin], [n1[0] - nmin, n2[0] - nmin], 'green')

    plt.xlabel('NORTH')
    plt.ylabel('EAST')

    plt.show()

def a_star(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])

    return path[::-1], path_cost

def probabilistic_roadmap(data, grid, grid_start, grid_goal, debug):

    ## Step 2 - Sample the Points
    nodes, polygons = sample_nodes(data, grid_start, grid_goal,debug)

    # Step 3 - Connect the Nodes and return the Graph
    g = create_graph(nodes, NUM_NODES, polygons, debug)

    path, path_cost =  a_star(g, heuristic, grid_start, grid_goal)
    if debug:
        print("Number of edges", len(g.edges))
        print("The Number of steps in the path is:{}".format(len(path)))
        print("The total cost of the path is: {0:5.2f}".format(path_cost))
    return path, path_cost