import numpy as np
import time
from collections import Counter
from ant_solution_ABW import ant_solution_best_worst

def ABW(adj_matrix, start_node, end_node, num_ants, evaporation_rate, max_epochs, initial_pheromone, alpha, beta):
    """
    Ant Colony Optimization with Best-Worst Ant System.

    Parameters:
    adj_matrix: adjacency matrix where each arc (row i, column j) has the cost
    start_node: root node (ant nest)
    end_node: destination
    num_ants: number of ants to perform the experiment
    evaporation_rate: pheromone evaporation rate between [0,1]
    max_epochs: maximum number of epochs
    initial_pheromone: initial pheromone level on all arcs
    alpha and beta: weigh the importance of the heuristic and pheromone values

    Returns:
    optimal_path: list containing the sequence of optimal nodes
    global_best_cost: total distance from start to end
    exec_time: execution time
    epochs: number of epochs performed
    """

    start_time = time.time()
    n, m = adj_matrix.shape
    pheromone_matrix = initial_pheromone + np.zeros((n, m))
    paths = [None] * num_ants  # Paths taken by each ant
    distances = np.zeros(num_ants)  # To order solutions from best to worst and check how many ants follow the same path
    threshold = 0  # Used as a parameter in pheromone mutation calculation
    iteration_counter = 0  # Used as a parameter in pheromone mutation calculation

    global_best_cost = float('inf')
    stagnant_count = 0  # Counts how many epochs the best solution stays the same
    epochs = 0
    consecutive_same_path_count = 0

    while consecutive_same_path_count < num_ants:
        for h in range(num_ants):  # Each ant makes its journey
            paths[h] = find_ant_solution(adj_matrix, pheromone_matrix, start_node, end_node, alpha, beta)
            distances[h] = paths[h][-1]

        # Global pheromone evaporation
        pheromone_matrix *= (1 - evaporation_rate)

        # Sort results to find the best and worst ant
        sorted_indices = np.argsort(distances)
        global_best_cost = paths[sorted_indices[0]][-1]

        # Deposit pheromone on the paths of the 2 best ants
        for best in range(2):
            if paths[sorted_indices[best]][-1] != float('inf'):
                for i in range(len(paths[sorted_indices[best]]) - 2):
                    pheromone_matrix[paths[sorted_indices[best]][i+1], paths[sorted_indices[best]][i]] += (
                        evaporation_rate * (1 / paths[sorted_indices[best]][-1])
                    )
                    if best == 0:
                        threshold += pheromone_matrix[paths[sorted_indices[best]][i+1], paths[sorted_indices[best]][i]]

        # Evaporate pheromone on the path of the worst ant
        worst_index = sorted_indices[-1]
        for i in range(len(paths[worst_index]) - 2):
            if not (paths[worst_index][i] in paths[sorted_indices[0]] and paths[worst_index][i+1] in paths[sorted_indices[0]]):
                pheromone_matrix[paths[worst_index][i+1], paths[worst_index][i]] *= (1 - evaporation_rate)

        # Pheromone mutation
        threshold /= (len(paths[sorted_indices[0]]) - 2)
        for j in range(1, num_ants - 1):
            for i in range(len(paths[sorted_indices[j]]) - 2):
                mutation = ((epochs - iteration_counter) / (max_epochs - iteration_counter)) * np.random.rand() * threshold
                if np.random.rand() < 0.5:
                    pheromone_matrix[paths[sorted_indices[j]][i+1], paths[sorted_indices[j]][i]] += mutation
                else:
                    pheromone_matrix[paths[sorted_indices[j]][i+1], paths[sorted_indices[j]][i]] -= mutation
                    if pheromone_matrix[paths[sorted_indices[j]][i+1], paths[sorted_indices[j]][i]] < 0:
                        pheromone_matrix[paths[sorted_indices[j]][i+1], paths[sorted_indices[j]][i]] = 0

        # Check algorithm termination criteria
        distances = distances[distances != float('inf')]  # Paths of lost ants don't count
        if distances.size > 0:
            most_common_distance, count = Counter(distances).most_common(1)[0]
            consecutive_same_path_count = count

        if most_common_distance != global_best_cost:
            stagnant_count += 1
        else:
            stagnant_count = 0

        if stagnant_count == 8 or epochs == max_epochs:
            iteration_counter = epochs
            consecutive_same_path_count = 0
            pheromone_matrix = initial_pheromone + np.zeros((n, m))
            # Deposit pheromone on the path of the best ant
            for i in range(len(paths[sorted_indices[0]]) - 2):
                pheromone_matrix[paths[sorted_indices[0]][i+1], paths[sorted_indices[0]][i]] += (
                    evaporation_rate * (1 / paths[sorted_indices[0]][-1])
                )

        epochs += 1

    # Return the optimal path
    optimal_path = paths[0][:-1]
    global_best_cost = paths[0][-1]
    exec_time = time.time() - start_time

    return optimal_path, global_best_cost, exec_time, epochs
