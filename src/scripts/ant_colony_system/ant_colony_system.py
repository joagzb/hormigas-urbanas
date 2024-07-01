import numpy as np
import time
from ant_solution_ACS import ant_solution_ACS

def ACS(adj_matrix, start_node, end_node, ants_number, global_evap_rate, local_evap_rate, transition_prob, max_epochs, initial_pheromone_matrix, pheromone_weight, heuristic_weight):
    """
    ANT COLONY SYSTEM. The ant system with elitism considers only a specific ant, the one that
    generated the best global solution. This ant will be the one that deposits the most pheromone.
    The other ants also apply a local update to the pheromone trails on their path, although much
    smaller compared to the best ant. This approach offers a balance between exploration and
    exploitation of accumulated knowledge. It is modified to explicitly allow exploration.
    The rule used is called the pseudo-random proportional rule.

    Parameters:
    adj_matrix : numpy.ndarray
        Adjacency matrix where adj_matrix[i, j] represents the cost between nodes i and j.
    start : int
        Starting node (nest).
    end : int
        Destination node (food source).
    nro_ants : int
        Number of ants (iterations) to perform.
    global_evap_rate : float
        Global pheromone evaporation rate in [0, 1].
    local_evap_rate : float
        Local pheromone evaporation rate in [0, 1] (exclusive to ACS).
    transition_prob : float
        Random transition probability parameter in [0, 1] (exclusive to ACS).
    max_epochs : int
        Maximum number of epochs (iterations) to run the algorithm.
    initial_pheromone : float
        Initial pheromone level on all edges.
    pheromone_weight : float
        Importance of pheromone information.
    heuristic_weight : float
        Importance of heuristic information.

    Returns:
    list, float, float, int : Optimal path, total distance of the optimal path, execution time, number of epochs executed.
    """

    tic = time.time()
    n, m = adj_matrix.shape
    pheromono_matrix = initial_pheromone_matrix + np.zeros((n, m))
    paths = [None] * ants_number  # Paths taken by each ant
    distances = np.zeros(ants_number)  # Path distances

    epochs = 0
    ant_count = 0

    while ant_count < ants_number:
        # Each ant makes its journey
        for ant in range(ants_number):
            paths[ant] = ant_solution_ACS(adj_matrix, pheromono_matrix, start_node, end_node, transition_prob, pheromone_weight, heuristic_weight)
            distances[ant] = paths[ant][-1]  # Last element is the total distance

            # Perform local pheromone update on the ant's path
            for i in range(len(paths[ant])-2):
                pheromono_matrix[paths[ant][i+1], paths[ant][i]] = (1 - local_evap_rate) * pheromono_matrix[paths[ant][i+1], paths[ant][i]] + local_evap_rate * initial_pheromone_matrix

        # Sort ants based on path distances
        sorted_indices = np.argsort(distances)

        # Global pheromone evaporation
        pheromono_matrix = (1 - global_evap_rate) * pheromono_matrix

        # Deposit pheromone on the paths of the best ants
        for best in sorted_indices[:2]:
            if paths[best][-1] != float('inf'):
                for i in range(len(paths[best])-2):
                    pheromono_matrix[paths[best][i+1], paths[best][i]] += global_evap_rate * (1 / paths[best][-1])

        # Check termination criterion
        distances = distances[distances != float('inf')]
        if len(distances) > 0:
            _, ant_count = np.mode(distances)

        # Handle stagnation conditions
        if epochs == max_epochs:
            pheromono_matrix = initial_pheromone_matrix + np.zeros((n, m))
            max_epochs *= 2

        epochs += 1

    optimal_path = paths[0][:-1]  # Optimal path sequence
    total_distance = paths[0][-1]  # Total distance of the optimal path
    execution_time = time.time() - tic  # Execution time

    return optimal_path, total_distance, execution_time, epochs
