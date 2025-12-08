import numpy as np
from time import time
from statistics import mode
from .ant_solution_MAXMIN import ant_solution_MAXMIN

def ACS_MAXMIN(adj_matrix, start_node, end_node, num_ants, p, q0, max_epochs, initial_pheromone, alpha, beta, max_restarts=3):
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
    max_restarts: Maximum number of pheromone resets before giving up (default: 3)

    Returns:
    path: Optimal path found by the algorithm
    costo_MejorGlobal: Total distance of the optimal path
    t: Time taken by the algorithm to complete
    epocas: Number of epochs executed
    """
    tic = time()

    n, m = adj_matrix.shape
    pheromone_matrix = initial_pheromone + np.zeros((n, m))  # Pheromone matrix initialized

    ant_paths = [None] * num_ants
    ant_distances = np.zeros(num_ants)
    
    # Track best solution found across all restarts
    global_best_path = None
    global_best_distance = float('inf')
    
    restarts = 0
    epochs = 0
    number_ants_following_path = 0
    
    while number_ants_following_path < num_ants:

        # Each ant performs its tour
        for ant_idx in range(num_ants):
            ant_paths[ant_idx] = ant_solution_MAXMIN(adj_matrix, pheromone_matrix, start_node, end_node, q0, alpha, beta)
            ant_distances[ant_idx] = ant_paths[ant_idx][-1]  # Last element is the total distance

            # Online evaporation and pheromone deposition (proposed by ACS)
            for i in range(len(ant_paths[ant_idx]) - 2):
                pheromone_matrix[ant_paths[ant_idx][i], ant_paths[ant_idx][i+1]] += p * initial_pheromone

        # Global pheromone evaporation
        pheromone_matrix *= (1 - p)

        # Sort results to find the best and worst ant paths
        sorted_indices = np.argsort(ant_distances)

        # Calculate maximum pheromone limit
        f_max = initial_pheromone # Default value
        if ant_paths[sorted_indices[0]][-1] != float('inf'):
            f_max = p * ant_paths[sorted_indices[0]][-1]
            
            # Track global best solution
            if ant_paths[sorted_indices[0]][-1] < global_best_distance:
                global_best_distance = ant_paths[sorted_indices[0]][-1]
                global_best_path = ant_paths[sorted_indices[0]][:]

        # Deposit pheromone on the paths of the 3 best ants
        for best in range(3):
            if ant_paths[sorted_indices[best]][-1] != float('inf'):
                for i in range(len(ant_paths[sorted_indices[best]]) - 2):
                    pheromone_matrix[ant_paths[sorted_indices[best]][i], ant_paths[sorted_indices[best]][i+1]] += p * (1 / ant_paths[sorted_indices[best]][-1])

        # Maintain minimum and maximum pheromone levels across the graph
        pheromone_matrix[pheromone_matrix < initial_pheromone] = initial_pheromone
        pheromone_matrix[pheromone_matrix > f_max] = f_max

        # Check stopping criterion of the algorithm
        valid_distances = ant_distances[ant_distances != float('inf')]  # Paths lost by ants do not count
        if len(valid_distances) > 0:
            most_common_distance = mode(valid_distances)
            number_ants_following_path = list(valid_distances).count(most_common_distance)

        # Correct stagnation conditions with restart limit
        if epochs == max_epochs:
            restarts += 1
            if restarts >= max_restarts:
                # Max restarts reached, exit with best solution found
                if global_best_path is not None:
                    path = global_best_path[:-1]  # Exclude the last element which is the total distance
                    costo_MejorGlobal = global_best_path[-1]
                else:
                    # No valid solution found at all
                    path = [start_node]
                    costo_MejorGlobal = float('inf')
                t = time() - tic
                return path, costo_MejorGlobal, t, epochs
            
            # Reset and try again
            pheromone_matrix = initial_pheromone + np.zeros((n, m))
            max_epochs *= 2
            number_ants_following_path = 0  # Reset convergence counter

        epochs += 1

    # Return the optimal path
    path = ant_paths[0][:-1]  # Exclude the last element which is the total distance
    costo_MejorGlobal = ant_paths[0][-1]  # Total distance of the optimal path
    t = time() - tic
    return path, costo_MejorGlobal, t, epochs
