import numpy as np
from ..utils.roulette_selection import roulette_wheel_selection

def ant_solution_MAXMIN(adj_matrix, pheromone_matrix, start_node, end_node, q0, alpha, beta):
    """
    Finds a solution path for an ant using the MAX-MIN Ant System.

    Parameters:
    adj_matrix: Adjacency matrix where each arc (row i, column j) has the cost
    pheromone_matrix: Pheromone matrix for each edge
    start_node: Root node (ant nest)
    end_node: Destination node (food)
    q0: Random state transition parameter between [0,1]
    alpha and beta: Weigh the importance of the heuristic and pheromone values

    Returns:
    path: Solution (path) found by the ant, including total cost at the end
    """
    path = [start_node]

    while path[0] != end_node:
        current_node = path[0]
        neighbors = np.where(adj_matrix[current_node, :] != 0)[0]  # Find neighboring nodes
        neighbors = neighbors[~np.isin(neighbors, path)]  # Do not choose nodes that have already been visited

        # The ant got lost. Stop the search
        if neighbors.size == 0:
            path.append(float('inf'))
            break

        # Probabilistic choice of the next node (proposed by Ant Colony System ACS)
        q = np.random.rand()
        if q <= q0:
            attractiveness = (pheromone_matrix[current_node, neighbors] ** alpha) * ((1.0 / adj_matrix[current_node, neighbors]) ** beta)
            next_node = neighbors[np.argmax(attractiveness)]
            path.insert(0, next_node)
        else:
            total_attractiveness = np.sum((pheromone_matrix[current_node, neighbors] ** alpha) * ((1.0 / adj_matrix[current_node, neighbors]) ** beta))
            probabilities = ((pheromone_matrix[current_node, neighbors] ** alpha) * ((1.0 / adj_matrix[current_node, neighbors]) ** beta)) / total_attractiveness
            next_node = neighbors[roulette_wheel_selection(probabilities)]
            path.insert(0, next_node)

    # return the path and calculate the total cost incurred
    if path[-1] != float('inf'):
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += adj_matrix[path[i+1], path[i]]
        path.append(total_cost)

    return path
