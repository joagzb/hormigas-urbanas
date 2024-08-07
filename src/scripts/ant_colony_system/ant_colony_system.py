import numpy as np
from time import time
from collections import Counter
from ant_solution_ACS import ant_solution_ACS
from utils.generators import generate_pheromone_map

def ACS(graph_map, start_node, end_node, ants_number, global_evap_rate, local_evap_rate, transition_prob, initial_pheromone_lvl, heuristic_weight, pheromone_weight):
    """
    ANT COLONY SYSTEM. The ant system with elitism considers only a specific ant, the one that
    generated the best global solution. This ant will be the one that deposits the most pheromone.
    The other ants also apply a local update to the pheromone trails on their path, although much
    smaller compared to the best ant. This approach offers a balance between exploration and
    exploitation of accumulated knowledge. It is modified to explicitly allow exploration.
    The rule used is called the pseudo-random proportional rule.

    Parameters:
    adj_matrix : numpy.ndarray
        Adjacency matrix where adj_matrix[i, j] represents the cost between nodes i and j.
    start : int
        Starting node (nest).
    end : int
        Destination node (food source).
    nro_ants : int
        Number of ants (iterations) to perform.
    global_evap_rate : float
        Global pheromone evaporation rate in [0, 1].
    local_evap_rate : float
        Local pheromone evaporation rate in [0, 1] (exclusive to ACS).
    transition_prob : float
        Random transition probability parameter in [0, 1] (exclusive to ACS).
    max_epochs : int
        Maximum number of epochs (iterations) to run the algorithm.
    initial_pheromone : float
        Initial pheromone level on all edges.
    pheromone_weight : float
        Importance of pheromone information.
    heuristic_weight : float
        Importance of heuristic information.

    Returns:
    Optimal path: list, total distance of the optimal path: float, execution time: float, number of epochs executed: int.
    """

    pheromone_graph = generate_pheromone_map(graph_map,initial_pheromone_lvl)
    routes = [None] * ants_number  # Paths taken by each ant
    distances = np.zeros(ants_number)

    epochs = 0
    counter = 0

    start_time = time()
    while counter < ants_number:
        # Each ant makes its journey
        for ant in range(ants_number):
            path_found, path_distance  = ant_solution_ACS(graph_map, pheromone_graph, start_node, end_node, transition_prob, heuristic_weight, pheromone_weight)
            routes[ant] = path_found
            distances[ant] = path_distance

        # Global pheromone evaporation
        for key in pheromone_graph:
            pheromone_graph[key] *= (1 - global_evap_rate)

        # Sort ants based on path distances
        sorted_indices_by_ant_solution = np.argsort(distances)
        best_ant = sorted_indices_by_ant_solution[0]

        # Perform local pheromone update on the ant's path
        for ant in range(ants_number):
            if distances[ant] != np.inf:
                for i in range(len(routes[ant])-1):
                    current_path_node = routes[ant][i]
                    next_path_node = routes[ant][i+1]
                    indx_next_node = graph_map["connections"][current_path_node].index(next_path_node)
                    pheromone_graph[current_path_node][indx_next_node] = ((1 - local_evap_rate) * pheromone_graph[current_path_node][indx_next_node]) + (local_evap_rate * (1 / distances[ant]))

                    if(best_ant == ant): # Deposit pheromone on the paths of the best ant
                        pheromone_graph[current_path_node][indx_next_node] = ((1 - global_evap_rate) * pheromone_graph[current_path_node][indx_next_node]) + (global_evap_rate * (1 / distances[best_ant]))

        # Analyze algorithm termination criteria
        number_of_solutions = distances[distances != np.inf].size
        if number_of_solutions > 0:
            _, counter = Counter(distances[distances != np.inf]).most_common(1)[0]

        epochs += 1

    optimal_path = routes[0]  # Optimal path sequence
    total_distance = distances[0]  # Total distance of the optimal path
    total_time = time() - start_time

    return optimal_path, total_distance, total_time, epochs