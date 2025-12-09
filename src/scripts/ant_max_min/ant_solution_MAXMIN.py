import numpy as np
from ..utils.roulette_selection import roulette_wheel_selection
from ..utils.heuristic_weights import normalize_for_selection


def ant_solution_MAXMIN(graph_map, pheromone_graph, start_node, end_node, q0, alpha, beta):
    """
    Finds a solution path for an ant using the MAX-MIN Ant System.

    Returns:
        tuple[list[int], float]: path and its cumulative cost (inf if no route).
    """
    path = [int(start_node)]
    max_steps = max(len(graph_map.get("node_index", [])), 50)

    while path[-1] != end_node:
        if len(path) > max_steps:
            return path, float('inf')
        current_node = path[-1]

        neighbors = np.array(graph_map["connections"].get(current_node, []))
        weights = np.array(graph_map["weights"].get(current_node, []))
        pheromones = np.array(pheromone_graph.get(current_node, []))

        if neighbors.size == 0:
            return path, float('inf')

        visited_mask = ~np.isin(neighbors, path)
        if visited_mask.any():
            valid_neighbors = neighbors[visited_mask]
            valid_weights = weights[visited_mask]
            valid_pheromones = pheromones[visited_mask]
        else:
            # All neighbors were already visited; allow a controlled revisit
            valid_neighbors = neighbors
            valid_weights = weights
            valid_pheromones = pheromones

        effective_weights = normalize_for_selection(valid_weights)
        with np.errstate(divide='ignore'):
            heuristic = (1.0 / effective_weights) ** beta

        attractiveness = (valid_pheromones ** alpha) * heuristic

        if np.random.rand() <= q0:
            best_idx = np.argmax(attractiveness)
            next_node = int(valid_neighbors[best_idx])
        else:
            total_attractiveness = np.sum(attractiveness)
            if total_attractiveness == 0:
                probabilities = np.ones_like(attractiveness) / len(attractiveness)
            else:
                probabilities = attractiveness / total_attractiveness
            selected_idx = roulette_wheel_selection(probabilities) - 1
            next_node = int(valid_neighbors[selected_idx])

        path.append(next_node)

    total_cost = 0.0
    for u, v in zip(path, path[1:]):
        try:
            idx = graph_map["connections"][u].index(v)
        except ValueError:
            return path, float('inf')
        total_cost += graph_map["weights"][u][idx]

    return path, total_cost
