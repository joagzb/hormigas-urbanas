import numpy as np
from time import time
from statistics import mode
from .ant_solution_MAXMIN import ant_solution_MAXMIN
from configuration.algorithm_settings import settings
from scripts.utils.generators import generate_pheromone_map

def ACS_MAXMIN(graph_map, start_node, end_node, num_ants, evaporation_rate, transition_probability, max_epochs, initial_pheromone, alpha, beta):
    """
    Ant Colony System with MAX-MIN strategy over a dict-based graph.

    Parameters:
    graph_map: Dict graph with keys
        - "node_index": set/list of nodes
        - "connections": dict[node] -> list of neighbor nodes (ordered)
        - "weights": dict[node] -> list of weights aligned to connections
    start_node: Root node (ant nest)
    end_node: Destination node (food)
    num_ants: Number of ants for the experiment
    evaporation_rate: Pheromone evaporation rate between [0,1]
    transition_probability: Random state transition parameter between [0,1]
    max_epochs: Maximum number of epochs to run the algorithm
    initial_pheromone: Initial pheromone level on all edges
    alpha and beta: Parameters to weigh the importance of heuristic and pheromone values

    Returns:
    total_epochs: Number of epochs executed
    """
    tic = time()

    pheromone_graph = generate_pheromone_map(graph_map, initial_pheromone)
    ant_paths = [None] * num_ants
    ant_distances = np.zeros(num_ants)

    epochs = 0
    number_ants_following_path = 0
    
    while number_ants_following_path < num_ants and epochs < max_epochs:
        # Each ant performs its tour
        for ant_idx in range(num_ants):
            ant_paths[ant_idx] = ant_solution_MAXMIN(graph_map, pheromone_graph, start_node, end_node, transition_probability, alpha, beta)
            ant_distances[ant_idx] = ant_paths[ant_idx][-1]

        # Global pheromone evaporation
        for node in pheromone_graph:
            pheromone_graph[node] *= (1 - evaporation_rate)

        # Sort results to find the best ant path
        sorted_indices = np.argsort(ant_distances)
        best_idx = sorted_indices[0]

        # Deposit pheromone on the best ant path only
        if ant_paths[best_idx][-1] != float('inf'):
            best_cost = ant_paths[best_idx][-1]
            for i in range(len(ant_paths[best_idx]) - 2):
                u = ant_paths[best_idx][i]
                v = ant_paths[best_idx][i + 1]
                j = graph_map["connections"][u].index(v)
                pheromone_graph[u][j] += evaporation_rate * (1.0 / best_cost)

        # Clamp pheromone within [f_min, f_max]
        f_min = settings.get("f_min", 0.0)
        f_max = settings.get("f_max", 1.0)
        for node in pheromone_graph:
            pheromone_graph[node] = np.clip(pheromone_graph[node], f_min, f_max)

        # Check stopping criterion
        finite = ant_distances[ant_distances != float('inf')]
        if finite.size > 0:
            most_common_distance = mode(finite)
            number_ants_following_path = list(finite).count(most_common_distance)

        epochs += 1

    # Return best finite path
    finite_mask = ant_distances != float('inf')
    if np.any(finite_mask):
        best_idx = np.argmin(ant_distances[finite_mask])
        finite_indices = np.where(finite_mask)[0]
        sel = finite_indices[best_idx]
        path = ant_paths[sel][:-1]
        cost = ant_paths[sel][-1]
    else:
        path = ant_paths[0][:-1]
        cost = ant_paths[0][-1]

    t = time() - tic
    return path, cost, t, epochs
