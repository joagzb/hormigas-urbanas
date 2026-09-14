from typing import Hashable

import numpy as np

from ..utils.roulette_selection import roulette_wheel_selection
from ..utils.heuristic_weights import normalize_for_selection
from ..utils.generators import validate_graph


def ant_solution_best_worst(graph_map: dict, pheromone_graph: dict, start_node: Hashable, end_node: Hashable, heuristic_weight: float, pheromone_weight: float):
  """
  Finds a path from the start node to the end node using an ant-inspired algorithm that incorporates pheromone levels
  and heuristic information to guide the search.

  Parameters:
  - graph_map (dict): A dictionary with the following keys:
      - "connections" (dict): Mapping of nodes to their connected neighbors. Each key is a node, and each value is a list of neighboring nodes.
      - "weights" (dict): Mapping of nodes to the weights of the edges leading to their neighbors. Each key is a node, and each value is a list of corresponding edge weights.
  - pheromone_graph (dict): A dictionary where keys are nodes and values are lists of pheromone levels for edges leading to neighbors.
  - start_node: The starting node (ant nest) in the graph.
  - end_node: The destination node (food) in the graph.
  - heuristic_weight (float): Legacy positional name for alpha, the pheromone exponent.
  - pheromone_weight (float): Legacy positional name for beta, the inverse-cost exponent.

  Returns:
  - solution_path (list): The sequence of nodes representing the path found by the ant. Includes `np.inf` if no valid path is found.
  - solution_cost (float): The total cost of the path found. Returns `np.inf` if the path is invalid or if the ant gets lost.

  Notes:
  - The function uses a probabilistic approach to select the next node based on pheromone levels and heuristic information.
  - The roulette wheel selection is employed to choose the next node based on calculated probabilities.
  - If the ant cannot move to any new node (i.e., all neighbors are visited or no valid path), it appends `np.inf` to indicate failure.
  """

  alpha = heuristic_weight
  beta = pheromone_weight
  require_edge_types = bool(graph_map.get('buses')) or any(isinstance(node, str) and node.startswith('bus:') for node in graph_map.get('node_index', []))
  validate_graph(graph_map, require_edge_types=require_edge_types)
  if start_node not in graph_map['node_index'] or end_node not in graph_map['node_index']:
    return None, np.inf
  solution_path = [start_node]
  visited_nodes = {start_node}
  solution_cost = 0

  # Construct a route without revisiting nodes
  while solution_path[-1] != end_node:
    current_node = solution_path[-1]
    neighbors = np.array(graph_map['connections'][current_node], dtype=object)
    neighbors_weights = np.array(graph_map['weights'][current_node])
    neighbors_edge_types = np.array(graph_map.get('edge_types', {}).get(current_node, ['walk'] * len(neighbors)), dtype=object)
    neighbors_pheromones = np.array(pheromone_graph[current_node])

    # Filter out visited nodes
    filter_visited_nodes_mask = np.array([neighbor not in visited_nodes for neighbor in neighbors], dtype=bool)
    neighbors = neighbors[filter_visited_nodes_mask]
    neighbors_weights = neighbors_weights[filter_visited_nodes_mask]
    neighbors_edge_types = neighbors_edge_types[filter_visited_nodes_mask]
    neighbors_pheromones = neighbors_pheromones[filter_visited_nodes_mask]

    # The ant gets lost if there are no unvisited neighbors
    if len(neighbors) == 0:
      solution_path.append(np.inf)
      break

    # Calculate the selection probabilities for each neighboring node
    pheromone_values = neighbors_pheromones**alpha
    selection_weights = normalize_for_selection(neighbors_weights, neighbors_edge_types)
    heuristic_values = (1.0 / selection_weights) ** beta
    combined = pheromone_values * heuristic_values
    sum_values = np.sum(combined)

    # Guard against degenerate probabilities
    if sum_values <= 0 or not np.isfinite(sum_values):
      next_node = neighbors[np.argmin(selection_weights)]
      solution_path.append(next_node)
      visited_nodes.add(next_node)
      continue

    probabilities = combined / sum_values

    # Select the next node using roulette wheel selection
    next_node_index = roulette_wheel_selection(probabilities)
    next_node = neighbors[next_node_index - 1]
    solution_path.append(next_node)
    visited_nodes.add(next_node)

  # Calculate the cost of the found path
  if solution_path[-1] != np.inf:
    for i in range(len(solution_path) - 1):
      neighbor_selected_index = graph_map['connections'][solution_path[i]].index(solution_path[i + 1])
      solution_cost += graph_map['weights'][solution_path[i]][neighbor_selected_index]
  else:
    solution_cost = np.inf

  return solution_path, solution_cost
