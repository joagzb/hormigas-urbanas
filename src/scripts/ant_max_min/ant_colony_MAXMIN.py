import numpy as np
import time
from ant_solution_MAXMIN import ant_solution_MAXMIN

def ACS_MAXMIN(adj_matrix, start_node, end_node, num_ants, p, q0, max_epochs, initial_pheromone, alpha, beta):
    """
    Ant Colony System with MAX-MIN strategy.

    Parameters:
    adj_matrix: Adjacency matrix where each arc (row i, column j) has the cost
    start_node: Root node (ant nest)
    end_node: Destination node (food)
    num_ants: Number of ants for the experiment
    p: Pheromone evaporation rate between [0,1]
    q0: Random state transition parameter between [0,1]
    max_epochs: Maximum number of epochs to run the algorithm
    initial_pheromone: Initial pheromone level on all edges
    alpha and beta: Parameters to weigh the importance of heuristic and pheromone values

    Returns:
    path: Optimal path found by the algorithm
    costo_MejorGlobal: Total distance of the optimal path
    t: Time taken by the algorithm to complete
    epocas: Number of epochs executed
    """
    tic = time.time()

    n, m = adj_matrix.shape
    pheromone_matrix = initial_pheromone + np.zeros((n, m))  # Pheromone matrix initialized

    ant_paths = [None] * num_ants
    ant_distances = np.zeros(num_ants)

    epochs = 0
    number_ants_following_path = 0
    while number_ants_following_path < num_ants:

        # Each ant performs its tour
        for ant_idx in range(num_ants):
            ant_paths[ant_idx] = ant_solution_MAXMIN(adj_matrix, pheromone_matrix, start_node, end_node, q0, alpha, beta)
            ant_distances[ant_idx] = ant_paths[ant_idx][-1]  # Last element is the total distance

            # Online evaporation and pheromone deposition (proposed by ACS)
            for i in range(len(ant_paths[ant_idx]) - 2):
                pheromone_matrix[ant_paths[ant_idx][i+1], ant_paths[ant_idx][i]] += p * initial_pheromone

        # Global pheromone evaporation
        pheromone_matrix *= (1 - p)

        # Sort results to find the best and worst ant paths
        sorted_indices = np.argsort(ant_distances)

        # Calculate maximum pheromone limit
        if ant_paths[sorted_indices[0]][-1] != float('inf'):
            f_max = p * ant_paths[sorted_indices[0]][-1]

        # Deposit pheromone on the paths of the 3 best ants
        for best in range(3):
            if ant_paths[sorted_indices[best]][-1] != float('inf'):
                for i in range(len(ant_paths[sorted_indices[best]]) - 2):
                    pheromone_matrix[ant_paths[sorted_indices[best]][i+1], ant_paths[sorted_indices[best]][i]] += p * (1 / ant_paths[sorted_indices[best]][-1])

        # Maintain minimum and maximum pheromone levels across the graph
        pheromone_matrix[pheromone_matrix < initial_pheromone] = initial_pheromone
        pheromone_matrix[pheromone_matrix > f_max] = f_max

        # Check stopping criterion of the algorithm
        ant_distances = ant_distances[ant_distances != float('inf')]  # Paths lost by ants do not count
        if len(ant_distances) > 0:
            _, number_ants_following_path = np.mode(ant_distances)  # Most followed path by ants (and how many ants follow it)

        # Correct stagnation conditions
        if epochs == max_epochs:
            pheromone_matrix = initial_pheromone + np.zeros((n, m))
            max_epochs *= 2

        epochs += 1

    # Return the optimal path
    path = ant_paths[0][:-1]  # Exclude the last element which is the total distance
    costo_MejorGlobal = ant_paths[0][-1]  # Total distance of the optimal path
    t = time.time() - tic
    return path, costo_MejorGlobal, t, epochs
