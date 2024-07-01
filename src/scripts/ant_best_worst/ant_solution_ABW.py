import numpy as np
from utils.roulette_selection import roulette_wheel_selection

def ant_solution_best_worst(adj_matrix, pheromone_matrix, start_node, end_node, pheromone_weight, heuristic_weight):
    """

    Parameters:
    adj_matrix (np.array): Adjacency matrix where each arc (row i, column j) has the cost
    pheromone_matrix (np.array): Pheromone matrix for each edge
    start_node (int): Root node (ant nest)
    end_node (int): Destination node (food)
    pheromone_weight -> alpha (float): Importance of pheromone
    heuristic_weight -> beta (float): Importance of heuristic information

    Returns:
    list: Solution path found by the ant
    """

    path = [start_node]

    while path[0] != end_node:
        current_node = path[0]
        neighbors = np.where(adj_matrix[current_node, :] != 0)[0]  # Find neighboring nodes
        neighbors = neighbors[~np.isin(neighbors, path)]  # Exclude already visited nodes

        # The ant is lost, end the search
        if len(neighbors) == 0:
            path.append(float('inf'))
            break

        # Calculate the probabilities for each neighboring node
        probabilities_sum = np.sum((pheromone_matrix[current_node, neighbors] ** pheromone_weight) * ((1.0 / adj_matrix[current_node, neighbors]) ** heuristic_weight))
        probabilities = ((pheromone_matrix[current_node, neighbors] ** pheromone_weight) * ((1.0 / adj_matrix[current_node, neighbors]) ** heuristic_weight)) / probabilities_sum

        # Select the next node based on the calculated probabilities
        next_node_index = roulette_wheel_selection(probabilities)
        path.insert(0, neighbors[next_node_index])

    # If the ant is not lost, return the path and calculate the total cost
    if path[-1] != float('inf'):
        total_cost = 0
        for i in range(len(path) - 1):
            total_cost += adj_matrix[path[i+1], path[i]]
        path.append(total_cost)

    return path
