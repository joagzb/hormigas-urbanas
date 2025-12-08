import numpy as np
from ..utils.roulette_selection import roulette_wheel_selection
from ..utils.heuristic_weights import normalize_for_selection

def ant_solution_best_worst(graph_map: dict, pheromone_graph: dict, start_node: int, end_node: int, heuristic_weight: float, pheromone_weight: float):
    """
    Finds a path from the start node to the end node using an ant-inspired algorithm that incorporates pheromone levels
    and heuristic information to guide the search.

    Parameters:
    - graph_map (dict): A dictionary with the following keys:
        - "connections" (dict): Mapping of nodes to their connected neighbors. Each key is a node, and each value is a list of neighboring nodes.
        - "weights" (dict): Mapping of nodes to the weights of the edges leading to their neighbors. Each key is a node, and each value is a list of corresponding edge weights.
    - pheromone_graph (dict): A dictionary where keys are nodes and values are lists of pheromone levels for edges leading to neighbors.
    - start_node (int): The starting node (ant nest) in the graph.
    - end_node (int): The destination node (food) in the graph.
    - heuristic_weight (float): The weight for the heuristic information used to guide the search. Higher values prioritize heuristic information.
    - pheromone_weight (float): The weight for the pheromone information used to guide the search. Higher values prioritize pheromone levels.

    Returns:
    - solution_path (list of int): The sequence of nodes representing the path found by the ant. Includes `np.inf` if no valid path is found.
    - solution_cost (float): The total cost of the path found. Returns `np.inf` if the path is invalid or if the ant gets lost.

    Notes:
    - The function uses a probabilistic approach to select the next node based on pheromone levels and heuristic information.
    - The roulette wheel selection is employed to choose the next node based on calculated probabilities.
    - If the ant cannot move to any new node (i.e., all neighbors are visited or no valid path), it appends `np.inf` to indicate failure.
    """

    solution_path = [start_node]
    solution_cost = 0

    while solution_path[-1] != end_node:
        current_node = solution_path[-1]
        neighbors = np.array(graph_map["connections"][current_node])
        neighbors_weights = np.array(graph_map["weights"][current_node])
        neighbors_pheromones = np.array(pheromone_graph[current_node])

        # Filter out visited nodes
        filter_visited_nodes_mask = ~np.isin(neighbors, solution_path)
        neighbors = neighbors[filter_visited_nodes_mask]
        neighbors_weights = neighbors_weights[filter_visited_nodes_mask]
        neighbors_pheromones = neighbors_pheromones[filter_visited_nodes_mask]

        # The ant gets lost if there are no unvisited neighbors
        if len(neighbors) == 0:
            solution_path.append(np.inf)
            break

        # Calculate the selection probabilities for each neighboring node
        pheromone_values = neighbors_pheromones ** pheromone_weight
        effective_weights = normalize_for_selection(neighbors_weights)
        heuristic_values = (1.0 / effective_weights) ** heuristic_weight
        combined = pheromone_values * heuristic_values
        sum_values = np.sum(combined)

        # Guard against degenerate probabilities
        if sum_values <= 0 or not np.isfinite(sum_values):
            next_node = int(neighbors[np.argmin(neighbors_weights)])
            solution_path.append(next_node)
            continue

        probabilities = combined / sum_values

        # Select the next node using roulette wheel selection
        next_node_index = roulette_wheel_selection(probabilities)
        solution_path.append(int(neighbors[next_node_index-1]))

    # Calculate the cost of the found path
    if solution_path[-1] != np.inf:
        for i in range(len(solution_path) - 1):
            neighbor_selected_index = graph_map["connections"][solution_path[i]].index(solution_path[i + 1])
            solution_cost += graph_map["weights"][solution_path[i]][neighbor_selected_index]
    else:
        solution_cost = np.inf

    return solution_path, solution_cost
