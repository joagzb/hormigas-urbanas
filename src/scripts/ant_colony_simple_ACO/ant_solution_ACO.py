import numpy as np
from utils.roulette_selection import roulette_wheel_selection

def ant_solution_ACO(adj_matrix, pheromone_matrix, start_node, end_node, heuristic_weight, pheromone_weight):
    """

    Parameters:
    adj_matrix: Adjacency matrix where each arc (row i, column j) has the cost
    pheromone_matrix: Pheromone matrix for each edge
    start_node: Starting node (ant hill)
    end_node: Destination node (food)
    heuristic_weight: Importance of the heuristic function
    pheromone_weight: Importance of pheromone trails

    Returns:
    path: List of visited nodes containing the solution path found by the ant
    """

    solution_path = [start_node]

    while solution_path[0] != end_node:
        current_node = solution_path[0]
        neighbors = np.where(adj_matrix[current_node, :] != 0)[0]  # Find neighbors of the current node
        neighbors = neighbors[~np.isin(neighbors, solution_path)]  # Do not revisit nodes

        if neighbors.size == 0:
            solution_path.append(float('inf'))  # The ant is lost. Stop the search
            break

        # Calculate probabilities for moving to the next node
        pheromone_values = pheromone_matrix[current_node, neighbors] ** heuristic_weight
        heuristic_values = (1.0 / adj_matrix[current_node, neighbors]) ** pheromone_weight
        sum_values = np.sum(pheromone_values * heuristic_values)
        probabilities = (pheromone_values * heuristic_values) / sum_values

        # Select the next node based on the roulette wheel selection
        next_node_index = roulette_wheel_selection(probabilities)
        solution_path.insert(0, neighbors[next_node_index])

    if solution_path[-1] != float('inf'):  # If the ant is not lost, return the path and calculate the total cost
        for i in range(len(solution_path) - 1):
          total_cost = sum(adj_matrix[solution_path[i + 1], solution_path[i]])
          solution_path.append(total_cost)

    return solution_path