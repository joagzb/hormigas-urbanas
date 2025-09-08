import numpy as np
from ..utils.roulette_selection import roulette_wheel_selection

def ant_solution_MAXMIN(graph_map, pheromone_graph, start_node, end_node, q0, alpha, beta):
    """
    Finds a solution path for an ant using the MAX-MIN Ant System.

    Parameters:
    graph_map: Dict graph with "connections" and "weights"
    pheromone_graph: Dict of pheromones aligned to connections per node
    start_node: Root node (ant nest)
    end_node: Destination node (food)
    q0: Random state transition parameter between [0,1]
    alpha and beta: Weigh the importance of the heuristic and pheromone values

    Returns:
    path: Solution (path) found by the ant, including total cost at the end
    """
    path = [start_node]

    while path[-1] != end_node:
        current_node = path[-1]
        neighbors = np.array(graph_map["connections"][current_node])
        neighbors_weights = np.array(graph_map["weights"][current_node])
        neighbors_pheromones = np.array(pheromone_graph[current_node])

        # Do not choose nodes that have already been visited
        mask = ~np.isin(neighbors, path)
        neighbors = neighbors[mask]
        neighbors_weights = neighbors_weights[mask]
        neighbors_pheromones = neighbors_pheromones[mask]

        # The ant got lost. Stop the search
        if neighbors.size == 0:
            path.append(float('inf'))
            break

        # Probabilistic choice of the next node transition (proposed by Ant Colony System ACS)
        q = np.random.rand()
        if q <= q0:
            attractiveness = (neighbors_pheromones ** alpha) * ((1.0 / neighbors_weights) ** beta)
            # Guard: if non-finite or all zeros, pick cheapest neighbor by cost
            if not np.isfinite(attractiveness).all() or np.all(attractiveness == 0):
                next_node = int(neighbors[np.argmin(neighbors_weights)])
            else:
                next_node = int(neighbors[np.argmax(attractiveness)])
            path.append(next_node)
        else:
            combined = (neighbors_pheromones ** alpha) * ((1.0 / neighbors_weights) ** beta)
            total_attractiveness = np.sum(combined)
            if total_attractiveness <= 0 or not np.isfinite(total_attractiveness):
                next_node = int(neighbors[np.argmin(neighbors_weights)])
            else:
                probabilities = combined / total_attractiveness
                next_node_index = roulette_wheel_selection(probabilities)
                next_node = int(neighbors[next_node_index - 1])
            path.append(next_node)

    # return the path and calculate the total cost incurred
    if path[-1] != float('inf'):
        total_cost = 0.0
        for a, b in zip(path[:-1], path[1:]):
            idx = graph_map["connections"][a].index(b)
            total_cost += graph_map["weights"][a][idx]
        path.append(total_cost)

    return path
