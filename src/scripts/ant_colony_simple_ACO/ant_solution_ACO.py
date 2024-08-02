import numpy as np
from ..utils.roulette_selection import roulette_wheel_selection

def ant_solution_ACO(graph_map: dict, pheromone_graph:dict, start_node:int, end_node:int, heuristic_weight:float, pheromone_weight:float):
    """
    Executes the Ant Colony Optimization (ACO) algorithm to find a path from a start node to an end node in a graph.

    Parameters:
    -----------
    graph_map : dict
        A dictionary representing the graph structure, where:
        - "connections": A dict with keys as nodes and values as lists of neighboring nodes.
        - "weights": A dict with keys as nodes and values as lists of corresponding edge weights to neighboring nodes.

    pheromone_graph : dict
        A dictionary where keys are nodes and values are lists representing the pheromone levels on the edges to neighboring nodes.

    start_node : int
        The node where the ant starts its search (ant hill).

    end_node : int
        The node where the ant aims to reach (food).

    heuristic_weight : float
        The exponent applied to the pheromone levels, representing the importance of the pheromone trail in the decision process.

    pheromone_weight : float
        The exponent applied to the inverse of the weights (costs), representing the importance of the heuristic (desirability) in the decision process.

    Returns:
    --------
    path : list of int or float
        A list of nodes representing the solution path found by the ant. If the ant gets "lost" and cannot find a valid path, `float('inf')` is appended to the path.

    solution_cost : float
        The total cost associated with the solution path. If the ant gets lost, this value is `float('inf')`.
    """

    solution_path = [start_node]
    solution_cost = 0

    while solution_path[-1] != end_node:
        current_node = solution_path[-1]
        neighbors = np.array(graph_map["connections"][current_node])
        neighbors_weights = np.array(graph_map["weights"][current_node])
        neighbors_pheromones = np.array(pheromone_graph[current_node])

        filter_visited_nodes_mask = ~np.isin(neighbors, solution_path)
        neighbors = neighbors[filter_visited_nodes_mask]
        neighbors_weights = neighbors_weights[filter_visited_nodes_mask]
        neighbors_pheromones = neighbors_pheromones[filter_visited_nodes_mask]

        if len(neighbors) == 0:
            solution_path.append(float('inf'))  # The ant is lost. Stop the search
            break

        # Calculate probabilities for moving to the next node
        pheromone_values = neighbors_pheromones ** heuristic_weight
        heuristic_values = (1.0 / neighbors_weights) ** pheromone_weight
        sum_values = np.sum(pheromone_values * heuristic_values)
        probabilities = (pheromone_values * heuristic_values) / sum_values

        # Select the next node based on the roulette wheel selection
        next_node_index = roulette_wheel_selection(probabilities)
        solution_path.append(neighbors[next_node_index-1])

    if solution_path[-1] != float('inf'):  # If the ant is not lost, return the path and calculate the total cost
        for i in range(len(solution_path) - 1):
          neighbor_selected_index = graph_map["connections"][solution_path[i]].index(solution_path[i + 1])
          solution_cost += graph_map["weights"][solution_path[i]][neighbor_selected_index]
    else:
        solution_cost = float('inf')

    return solution_path,solution_cost