from typing import Hashable

import numpy as np

from ..utils.roulette_selection import roulette_wheel_selection
from ..utils.heuristic_weights import normalize_for_selection


def ant_solution_ACO(graph_map: dict, pheromone_graph: dict, start_node: Hashable, end_node: Hashable, heuristic_weight: float, pheromone_weight: float):
  """
  Executes the Ant Colony Optimization (ACO) algorithm to find a path from a start node to an end node in a graph.

  Parameters:
  -----------
  graph_map : dict
      A preflighted graph with opaque node IDs and positionally aligned
      ``connections``, ``weights``, and ``edge_types`` rows.

  pheromone_graph : dict
      A dictionary where keys are nodes and values are lists representing the pheromone levels on the edges to neighboring nodes.

  start_node
      The node where the ant starts its search (ant hill).

  end_node
      The node where the ant aims to reach (food).

  heuristic_weight : float
      Legacy positional name for alpha, the pheromone exponent.

  pheromone_weight : float
      Legacy positional name for beta, the inverse-cost exponent.

  Returns:
  --------
  path : list
      A list of nodes representing the solution path found by the ant. If the ant gets "lost" and cannot find a valid path, `float('inf')` is appended to the path.

  solution_cost : float
      The total cost associated with the solution path. If the ant gets lost, this value is `float('inf')`.
  """

  alpha = heuristic_weight
  beta = pheromone_weight
  solution_path = [start_node]
  visited_nodes = {start_node}
  solution_cost = 0

  # Construct a route without revisiting nodes
  while solution_path[-1] != end_node:
    current_node = solution_path[-1]
    neighbors = np.array(graph_map['connections'][current_node], dtype=object)
    neighbors_weights = np.array(graph_map['weights'][current_node])
    neighbors_edge_types = np.array(graph_map['edge_types'][current_node], dtype=object)
    neighbors_pheromones = np.array(pheromone_graph[current_node])

    filter_visited_nodes_mask = np.array([neighbor not in visited_nodes for neighbor in neighbors], dtype=bool)
    neighbors = neighbors[filter_visited_nodes_mask]
    neighbors_weights = neighbors_weights[filter_visited_nodes_mask]
    neighbors_edge_types = neighbors_edge_types[filter_visited_nodes_mask]
    neighbors_pheromones = neighbors_pheromones[filter_visited_nodes_mask]

    if len(neighbors) == 0:
      solution_path.append(np.inf)  # The ant is lost. Stop the search
      break

    # Calculate probabilities for moving to the next node
    pheromone_values = neighbors_pheromones**alpha
    selection_weights = normalize_for_selection(neighbors_weights, neighbors_edge_types)
    heuristic_values = (1.0 / selection_weights) ** beta
    combined = pheromone_values * heuristic_values
    sum_values = np.sum(combined)

    # guard against numerical issues (e.g., all zeros)
    if sum_values <= 0 or not np.isfinite(sum_values):
      # Fallback to greedy by cost
      next_node = neighbors[np.argmin(selection_weights)]
      solution_path.append(next_node)
      visited_nodes.add(next_node)
      continue

    probabilities = combined / sum_values

    # select the next node based on the roulette wheel selection
    next_node_index = roulette_wheel_selection(probabilities)
    next_node = neighbors[next_node_index]
    solution_path.append(next_node)
    visited_nodes.add(next_node)

  if solution_path[-1] != np.inf:  # If the ant is not lost, return the path and calculate the total cost
    for i in range(len(solution_path) - 1):
      neighbor_selected_index = graph_map['connections'][solution_path[i]].index(solution_path[i + 1])
      solution_cost += graph_map['weights'][solution_path[i]][neighbor_selected_index]
  else:
    solution_cost = np.inf

  return solution_path, solution_cost
