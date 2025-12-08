import numpy as np
from ..utils.roulette_selection import roulette_wheel_selection
from ..utils.heuristic_weights import normalize_for_selection

def ant_solution_MAXMIN(graph_map, pheromone_graph, start_node, end_node, q0, alpha, beta):
    """
    Finds a solution path for an ant using the MAX-MIN Ant System.

    Parameters:
    graph_map: Dictionary graph with keys "connections" and "weights"
    pheromone_graph: Pheromone dictionary for each node
    start_node: Root node (ant nest)
    end_node: Destination node (food)
    q0: Random state transition parameter between [0,1]
    alpha and beta: Weigh the importance of the heuristic and pheromone values

    Returns:
    path: Solution (path) found by the ant, including total cost at the end
    """
    path = [start_node]
    current_cost = 0

    while path[-1] != end_node:
        current_node = path[-1]
        
        # Get neighbors, weights, and pheromones
        neighbors = np.array(graph_map["connections"].get(current_node, []))
        weights = np.array(graph_map["weights"].get(current_node, []))
        pheromones = np.array(pheromone_graph.get(current_node, []))
        
        if neighbors.size == 0:
            path.append(float('inf'))
            break
            
        # Filter out visited nodes
        visited_mask = ~np.isin(neighbors, path)
        valid_neighbors = neighbors[visited_mask]
        valid_weights = weights[visited_mask]
        valid_pheromones = pheromones[visited_mask]
        
        if valid_neighbors.size == 0:
            path.append(float('inf'))
            break
            
        # Calculate attractiveness
        # Avoid division by zero in heuristic
        effective_weights = normalize_for_selection(valid_weights)
        with np.errstate(divide='ignore'):
                heuristic = (1.0 / effective_weights) ** beta
        
        attractiveness = (valid_pheromones ** alpha) * heuristic
        
        # Probabilistic choice (MMAS rule)
        if np.random.rand() <= q0:
            # Exploitation: best neighbor
            best_idx = np.argmax(attractiveness)
            next_node = valid_neighbors[best_idx]
        else:
            # Exploration: roulette wheel selection
            total_attractiveness = np.sum(attractiveness)
            if total_attractiveness == 0:
                    probabilities = np.ones_like(attractiveness) / len(attractiveness)
            else:
                    probabilities = attractiveness / total_attractiveness
            
            # roulette_wheel_selection returns 1-based index
            selected_idx = roulette_wheel_selection(probabilities) - 1
            next_node = valid_neighbors[selected_idx]

        path.append(next_node)

    # Calculate total cost
    if path[-1] != float('inf'):
        total_cost = 0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            # Find index of v in u's connections
            try:
                idx = graph_map["connections"][u].index(v)
                total_cost += graph_map["weights"][u][idx]
            except (ValueError, IndexError):
                path[-1] = float('inf') # Mark as invalid if edge not found
                return path
                
        path.append(total_cost)

    return path
