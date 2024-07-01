import numpy as np
import scipy.sparse
from time import time
from collections import Counter
from ant_solution_ACO import ant_solution_ACO

def ACO(adj_matrix, start_point, end_point, ants_number, evaporation_rate, initial_pheromone_lvl, heuristic_weight, pheromone_weight):
    """
    Simple Ant Colony Optimization (ACO)

    Parameters:
    adj_matrix: Adjacency matrix where each arc (row i, column j) has the cost
    start_point: Starting node (ant hill)
    end_point: Destination node (food)
    ants_number: Number of ants for the experiment
    evaporation_rate: Pheromone evaporation rate between [0,1]
    initial_pheromone_lvl: Initial pheromone level on all arcs
    heuristic_weight: Importance of heuristic function (heuristic_weight=0 ignores heuristic)
    pheromone_weight: Importance of pheromone trails (pheromone_weight=0 ignores pheromone trails)

    Returns:
    path: List containing the optimal sequence of nodes
    cost: Total cost of the optimal path found
    total_time: Time taken to solve the problem
    epochs: Number of epochs run
    """

    start_time = time()
    n, m = adj_matrix.shape
    pheromone = initial_pheromone_lvl + scipy.sparse.csr_matrix((n, m))  # Pheromone matrix
    routes = [None] * ants_number  # Paths taken by each ant
    distances = np.zeros(ants_number)  # To sort solutions from best to worst

    epochs = 0
    counter = 0

    while counter < ants_number:
        # Each ant makes its journey
        for h in range(ants_number):
            routes[h] = ant_solution_ACO(adj_matrix, pheromone, start_point, end_point, heuristic_weight, pheromone_weight)
            distances[h] = routes[h][-1]  # Last value in the path is the total cost

        # Global pheromone evaporation
        pheromone = (1 - evaporation_rate) * pheromone

        # Global pheromone deposition
        for h in range(ants_number):
            if distances[h] != np.inf:
                for i in range(len(routes[h]) - 2):
                    pheromone[routes[h][i + 1], routes[h][i]] += 1 / distances[h]

        # Analyze algorithm termination criteria
        distances = distances[distances != np.inf]  # Paths of lost ants don't count
        if distances.size > 0:
            most_common_distance, counter = Counter(distances).most_common(1)[0]

        epochs += 1

    # Return the optimal path
    path = routes[0][:-1]
    cost = routes[0][-1]
    total_time = time() - start_time

    return path, cost, total_time, epochs
