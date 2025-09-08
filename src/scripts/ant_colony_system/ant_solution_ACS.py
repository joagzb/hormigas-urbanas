import numpy as np
from ..utils.roulette_selection import roulette_wheel_selection

def ant_solution_ACS(graph_map: dict, pheromone_graph:dict, start_node:int, end_node:int, q0: float, heuristic_weight:float, pheromone_weight:float):
    """
    Ant Colony System (ACS) solution for a single ant traversing the graph to find a path.

    Parameters:
    adj_matrix: adjacency matrix where each arc (i, j) has the cost
    pheromone_matrix: pheromone matrix for each edge
    start_node: root node (ant nest)
    end_node: destination node
    q0: constant parameter for probabilistic transition (exclusive to ACS) between [0,1]
    pheromone_weight (alpha): importance of pheromone values
    heuristic_weight (beta): importance of heuristic values (1 / cost)

    Returns:
    path: solution path found by the ant
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

         # The ant is lost. Stop the search
        if len(neighbors) == 0:
            solution_path.append(np.inf)
            break

        # Probabilistic choice of the next node (proposed by Ant Colony System ACS)
        q = np.random.rand()
        if q <= q0:
            Z = (neighbors_pheromones ** heuristic_weight) * ((1.0 / neighbors_weights) ** pheromone_weight)
            # Guard: if Z are non-finite or all-zero, fall back to cheapest neighbor
            if not np.isfinite(Z).all() or np.all(Z == 0):
                next_node = int(neighbors[np.argmin(neighbors_weights)])
            else:
                next_node = int(neighbors[np.argmax(Z)])
            solution_path.append(next_node)
        else:
            pheromone_values = neighbors_pheromones ** heuristic_weight
            heuristic_values = (1.0 / neighbors_weights) ** pheromone_weight
            combined = pheromone_values * heuristic_values
            sum_values = np.sum(combined)
            if sum_values <= 0 or not np.isfinite(sum_values):
                next_node = int(neighbors[np.argmin(neighbors_weights)])
                solution_path.append(next_node)
            else:
                probabilities = combined / sum_values
                # Select the next node based on the roulette wheel selection
                next_node_index = roulette_wheel_selection(probabilities)
                solution_path.append(int(neighbors[next_node_index-1]))

    # return the path and calculate the incurred costs
    if solution_path[-1] != np.inf:
        for i in range(len(solution_path) - 1):
            neighbor_selected_index = graph_map["connections"][solution_path[i]].index(solution_path[i + 1])
            solution_cost += graph_map["weights"][solution_path[i]][neighbor_selected_index]
    else:
        solution_cost = np.inf

    return solution_path, solution_cost
