import numpy as np
from time import time
from collections import Counter
from .ant_solution_ABW import ant_solution_best_worst
from scripts.utils.generators import generate_pheromone_map
from configuration.algorithm_settings import settings


def ABW(graph_map, start_node, end_node, ants_number, global_evap_rate, max_epochs, initial_pheromone_lvl, heuristic_weight, pheromone_weight):
    """
    Perform Ant Colony Optimization using the Best-Worst Ant System (BWAS) to find the shortest path in a graph.

    Parameters:
    - graph_map (dict): Graph in dict form with:
        - "connections": dict[node] -> list of neighboring nodes (ordered)
        - "weights": dict[node] -> list of edge weights aligned with connections
        - "node_index": set/list of nodes
    - start_node (int): The starting node (ant nest) in the graph.
    - end_node (int): The destination node in the graph.
    - ants_number (int): The number of ants (agents) used to explore paths in the graph.
    - global_evap_rate (float): The rate at which pheromone evaporates on each edge, in the range [0, 1].
    - max_epochs (int): The maximum number of iterations (epochs) for the optimization process.
    - initial_pheromone_lvl (float): The initial pheromone level assigned to all edges in the graph.
    - heuristic_weight (float): The weight for the heuristic information in path decisions.
    - pheromone_weight (float): The weight for the pheromone trail information in path decisions.

    Returns:
    - optimal_path (list of int): The sequence of nodes representing the optimal path found.
    - global_best_cost (float): The total cost (distance) of the optimal path.
    - exec_time (float): The total execution time of the algorithm in seconds.
    - epochs (int): The number of epochs (iterations) performed during the optimization.

    Notes:
    - The function optimizes paths by having ants explore the graph, update pheromone levels, and utilize both pheromone and heuristic information.
    - Pheromone levels are updated based on the best and worst solutions found by the ants.
    - If the best solution stagnates for a number of epochs, the pheromone levels are reset, and the optimization continues.
    """

    pheromone_graph = generate_pheromone_map(graph_map, initial_pheromone_lvl)
    routes = [None] * ants_number  # Paths taken by each ant
    distances = np.zeros(ants_number)  # Distances of the paths found by each ant

    epoch_before_restart = 0  # Used for pheromone mutation calculation
    stagnant_count = 0  # Counter for epochs where the best solution remains unchanged
    max_stagnant_count = 8  # Threshold for stagnation before pheromone reset
    global_best_cost = np.inf  # Initial global best cost set to infinity
    min_pheromone_lvl = settings['f_min']  # Minimum pheromone level allowed on trails

    epochs = 0
    same_path_solution_counter = 0

    start_time = time()
    while same_path_solution_counter < ants_number and epochs < max_epochs:
        # Each ant finds a path
        for ant in range(ants_number):
            path_found, path_distance = ant_solution_best_worst(graph_map, pheromone_graph, start_node, end_node, heuristic_weight, pheromone_weight)
            routes[ant] = path_found
            distances[ant] = path_distance

        # Update global pheromone levels with evaporation
        for key in pheromone_graph:
            pheromone_graph[key] *= (1 - global_evap_rate)

        # Sort ants based on their path distances
        sorted_indices_by_ant_solution = np.argsort(distances)
        best_ant = sorted_indices_by_ant_solution[0]
        worst_ant = sorted_indices_by_ant_solution[-1]

        # Default threshold based on global pheromone mean (fallback)
        try:
            all_vals = np.concatenate([np.asarray(v) for v in pheromone_graph.values() if len(v) > 0])
            threshold = float(np.mean(all_vals)) if all_vals.size > 0 else float(min_pheromone_lvl)
        except Exception:
            threshold = float(min_pheromone_lvl)

        # Update pheromone on the best ant's path and compute threshold from it
        if distances[best_ant] != np.inf:
            best_ant_pheromone_trail = []
            for i in range(len(routes[best_ant]) - 1):
                current_path_node = routes[best_ant][i]
                next_path_node = routes[best_ant][i + 1]
                indx_next_node = graph_map["connections"][current_path_node].index(next_path_node)
                pheromone_graph[current_path_node][indx_next_node] += 1 / distances[best_ant]
                best_ant_pheromone_trail.append(pheromone_graph[current_path_node][indx_next_node])

            global_best_cost = distances[best_ant]
            if len(best_ant_pheromone_trail) > 0:
                threshold = float(np.mean(best_ant_pheromone_trail))

        # Build edge set for best ant to avoid penalizing best edges
        best_edges = set()
        if distances[best_ant] != np.inf:
            for i in range(len(routes[best_ant]) - 1):
                best_edges.add((routes[best_ant][i], routes[best_ant][i + 1]))

        # Evaporate pheromone on the worst ant's specific edges (not shared with best edges)
        # Guard against lost ants whose path ends with np.inf
        if distances[worst_ant] != np.inf and routes[worst_ant] is not None:
            for i in range(len(routes[worst_ant]) - 1):
                current_path_node = routes[worst_ant][i]
                next_path_node = routes[worst_ant][i + 1]
                # Skip invalid edges (lost ant marker) or edges shared with best path
                if (current_path_node is None or next_path_node is None or
                    current_path_node == np.inf or next_path_node == np.inf or
                    (current_path_node, next_path_node) in best_edges):
                    continue
                # Some routes may include edges not present in connections (defensive)
                neighbors = graph_map["connections"].get(current_path_node, [])
                try:
                    indx_next_node = neighbors.index(next_path_node)
                except ValueError:
                    continue
                pheromone_graph[current_path_node][indx_next_node] *= (1 - global_evap_rate)

        # Perform pheromone trail mutation
        denom = max(1, (max_epochs - epoch_before_restart))
        mutation = ((epochs - epoch_before_restart) / denom) * np.random.rand() * float(threshold)
        for trail in pheromone_graph.keys():
            if np.random.rand() < 0.5:
                pheromone_graph[trail] += mutation
            else:
                pheromone_graph[trail] -= mutation
                trail_values = pheromone_graph[trail]
                trail_values[trail_values < min_pheromone_lvl] = min_pheromone_lvl
                pheromone_graph[trail] = trail_values

        # Check termination criteria
        number_of_solutions = distances[distances != np.inf].size
        most_common_distance = None
        if number_of_solutions > 0:
            most_common_distance, same_path_solution_counter = Counter(distances[distances != np.inf]).most_common(1)[0]

        if (most_common_distance is not None) and (most_common_distance != global_best_cost):
            stagnant_count += 1

        if stagnant_count == max_stagnant_count:
            epoch_before_restart = epochs
            stagnant_count = 0
            pheromone_graph = generate_pheromone_map(graph_map, initial_pheromone_lvl)

        epochs += 1

    # Select best finite route if available
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
