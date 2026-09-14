from typing import Hashable

import numpy as np

from ..utils.heuristic_weights import normalize_weights_for_selection
from ..utils.roulette_selection import roulette_wheel_selection


def ant_solution_ACS(
  graph_map: dict,
  pheromone_graph: dict,
  start_node: Hashable,
  end_node: Hashable,
  q0: float,
  heuristic_weight: float,
  pheromone_weight: float,
  local_evap_rate: float = 0.0,
  initial_pheromone_lvl: float | None = None,
):
  """Build one ACS path while updating each selected edge immediately.

  Parameters:
    - graph_map (dict): Preflighted graph with opaque node IDs and aligned
        ``connections``, ``weights``, and ``edge_types`` mappings.

    - pheromone_graph (dict): Pheromone values aligned with graph edges.

    - start_node: Starting node.

    - end_node: Destination node.

    - q0 (float): Probability threshold for greedy selection.

    - heuristic_weight (float): Legacy positional name for alpha, the pheromone influence exponent.

    - pheromone_weight (float): Legacy positional name for beta, the inverse-cost influence exponent.

    - local_evap_rate (float): ACS local update rate (xi).

    - initial_pheromone_lvl (float | None): Baseline initial_pheromone_lvl for the local update, or ``None`` when the caller has no baseline value.

  Returns:
  --------
  solution_path : list
      A list of nodes representing the solution path found by the ant. If the ant gets "lost" and cannot find a valid path, `float('inf')` is appended to the path.

  solution_cost : float
      The total cost associated with the solution path. If the ant gets lost, this value is `float('inf')`.
  """

  alpha = heuristic_weight
  beta = pheromone_weight
  solution_path = [start_node]
  visited_nodes = {start_node}
  solution_cost = 0

  # Construct a route
  while solution_path[-1] != end_node:
    current_node = solution_path[-1]

    neighbors = np.array(graph_map['connections'][current_node], dtype=object)
    neighbors_weights = np.array(graph_map['weights'][current_node])
    neighbors_edge_types = np.array(graph_map['edge_types'][current_node], dtype=object)
    neighbors_pheromones = np.array(pheromone_graph[current_node])

    # Filter out visited nodes
    filter_visited_nodes_mask = np.array([neighbor not in visited_nodes for neighbor in neighbors], dtype=bool)
    neighbors = neighbors[filter_visited_nodes_mask]
    neighbors_weights = neighbors_weights[filter_visited_nodes_mask]
    neighbors_edge_types = neighbors_edge_types[filter_visited_nodes_mask]
    neighbors_pheromones = neighbors_pheromones[filter_visited_nodes_mask]

    if len(neighbors) == 0:
      solution_path.append(np.inf)  # The ant is lost. Stop the search.
      break

    # Probabilistic choice of the next node (proposed by Ant Colony System ACS)
    selection_weights = normalize_weights_for_selection(neighbors_weights, neighbors_edge_types)
    q = np.random.rand()

    if q <= q0:
      desirability = (neighbors_pheromones**alpha) * ((1.0 / selection_weights) ** beta)
      if not np.isfinite(desirability).all() or np.all(desirability == 0):
        next_node = neighbors[np.argmin(selection_weights)]
      else:
        next_node = neighbors[np.argmax(desirability)]
    else:
      # ACS exploration uses the AS distribution with alpha fixed at one.
      pheromone_values = neighbors_pheromones
      heuristic_values = (1.0 / selection_weights) ** beta
      combined = pheromone_values * heuristic_values
      sum_values = np.sum(combined)
      if sum_values <= 0 or not np.isfinite(sum_values):
        next_node = neighbors[np.argmin(selection_weights)]
      else:
        probabilities = combined / sum_values
        next_node_index = roulette_wheel_selection(probabilities)
        next_node = neighbors[next_node_index]

    solution_path.append(next_node)
    visited_nodes.add(next_node)

    # Local pheromone update during route construction
    if initial_pheromone_lvl is not None and local_evap_rate:
      edge_index = graph_map['connections'][current_node].index(next_node)
      current_pheromone = pheromone_graph[current_node][edge_index]
      pheromone_graph[current_node][edge_index] = (1 - local_evap_rate) * current_pheromone + local_evap_rate * initial_pheromone_lvl

  # Calculate the cost of the found path
  if solution_path[-1] != np.inf:
    for i in range(len(solution_path) - 1):
      neighbor_selected_index = graph_map['connections'][solution_path[i]].index(solution_path[i + 1])
      solution_cost += graph_map['weights'][solution_path[i]][neighbor_selected_index]
  else:
    solution_cost = np.inf

  return solution_path, solution_cost
