import numpy as np
from time import time
from collections import Counter
from scripts.utils.generators import generate_pheromone_map
from .ant_solution_ACO import ant_solution_ACO

def ACO(graph_map, start_node, end_node, ants_number, evaporation_rate, initial_pheromone_lvl, heuristic_weight, pheromone_weight, max_epochs: int = 500):
    """
    Performs Simple Ant Colony Optimization (ACO) to find the optimal path between start and end nodes in a graph.

    Parameters:
    -----------
    graph_map : dict
        A dictionary representing the graph structure, where:
        - "connections": A dict with keys as nodes and values as lists of neighboring nodes.
        - "weights": A dict with keys as nodes and values as lists of corresponding edge weights to neighboring nodes.

    start_node : int
        The starting node (ant hill) where all ants begin their journey.

    end_node : int
        The destination node (food) where all ants are trying to reach.

    ants_number : int
        The number of ants used in the simulation.

    evaporation_rate : float
        The rate at which pheromone evaporates after each epoch, must be in the range [0, 1].

    initial_pheromone_lvl : float
        The initial level of pheromone on all paths, used to initialize the pheromone graph.

    heuristic_weight : float
        The exponent applied to the pheromone levels, determining the importance of pheromone trails in the decision process.

    pheromone_weight : float
        The exponent applied to the inverse of the weights (costs), determining the importance of heuristic desirability in the decision process.

    Returns:
    --------
    path : list of int
        The list of nodes representing the optimal sequence of nodes found by the ants.

    cost : float
        The total cost associated with the optimal path found by the ants.

    total_time : float
        The time taken (in seconds) to complete the optimization process.

    epochs : int
        The number of epochs (iterations) the algorithm ran before converging to a solution.
    """
    start_time = time()
    
    pheromone_graph = generate_pheromone_map(graph_map, initial_pheromone_lvl)
    routes = [None] * ants_number
    distances = np.zeros(ants_number)

    epochs = 0
    counter = 0
    
    while counter < ants_number and epochs < max_epochs:
        # Each ant makes its journey
        for ant in range(ants_number):
            path_found,path_distance = ant_solution_ACO(graph_map, pheromone_graph, start_node, end_node, heuristic_weight, pheromone_weight)
            routes[ant] = path_found
            distances[ant] = path_distance

        # Global pheromone evaporation
        for key in pheromone_graph:
            pheromone_graph[key] *= (1 - evaporation_rate)

        # Global pheromone deposition
        for ant in range(ants_number):
            if distances[ant] != np.inf:
                for i in range(len(routes[ant]) - 1):
                    current_route_node = routes[ant][i]
                    next_route_node = routes[ant][i+1]
                    indx_next_node = graph_map["connections"][current_route_node].index(next_route_node)
                    pheromone_graph[current_route_node][indx_next_node] += 1 / distances[ant]

        # Check termination criteria
        number_of_solutions = distances[distances != np.inf].size
        if number_of_solutions > 0:
            _, counter = Counter(distances[distances != np.inf]).most_common(1)[0]

        epochs += 1
    
    # Return the optimal path
    finite_mask = distances != np.inf
    if np.any(finite_mask):
        best_idx = np.argmin(distances[finite_mask])
        finite_indices = np.where(finite_mask)[0]
        selected = finite_indices[best_idx]
        optimal_path = routes[selected]
        total_distance = distances[selected]
    else:
        optimal_path = routes[0]
        total_distance = distances[0]
    
    total_time = time() - start_time

    return optimal_path, total_distance, total_time, epochs
