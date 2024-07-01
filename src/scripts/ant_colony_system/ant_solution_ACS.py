import numpy as np
from utils.roulette_selection import roulette_wheel_selection

def ant_solution_ACS(adj_matrix, pheromone_matrix, start_node, end_node, q0, pheromone_weight, heuristic_weight):
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

    path = [start_node]

    while path[0] != end_node:
        current_node = path[0]
        neighbors = np.where(adj_matrix[current_node, :] != 0)[0]  # Find neighboring nodes of the current node
        neighbors = neighbors[~np.isin(neighbors, path)]  # Do not choose nodes that have already been visited

         # The ant is lost. Stop the search
        if len(neighbors) == 0:
            path.append(float('inf'))
            break

        # Probabilistic choice of the next node (proposed by Ant Colony System ACS)
        q = np.random.rand()
        if q <= q0:
            Z = (pheromone_matrix[current_node, neighbors] ** pheromone_weight) * ((1.0 / adj_matrix[current_node, neighbors]) ** heuristic_weight)
            next_node = neighbors[np.argmax(Z)]
            path.insert(0, next_node)
        else:
            total = np.sum((pheromone_matrix[current_node, neighbors] ** pheromone_weight) * ((1.0 / adj_matrix[current_node, neighbors]) ** heuristic_weight))
            probabilities = ((pheromone_matrix[current_node, neighbors] ** pheromone_weight) * ((1.0 / adj_matrix[current_node, neighbors]) ** heuristic_weight)) / total
            next_node = neighbors[roulette_wheel_selection(probabilities)]
            path.insert(0, next_node)

    # return the path and calculate the incurred costs
    if path[-1] != float('inf'):
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += adj_matrix[path[i + 1], path[i]]
        path.append(total_cost)

    return path