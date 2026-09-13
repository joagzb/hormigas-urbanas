from typing import Hashable

import numpy as np

from ..utils.roulette_selection import roulette_wheel_selection
from ..utils.heuristic_weights import normalize_for_selection
from ..utils.generators import validate_graph


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
      graph_map (dict): Graph with ``node_index``, ``connections``, and
          ``weights`` mappings.
      pheromone_graph (dict): Pheromone values aligned with graph edges.
      start_node: Starting node.
      end_node: Destination node.
      q0 (float): Probability threshold for greedy selection.
      heuristic_weight (float): Legacy positional name for alpha,
          the pheromone influence exponent.
      pheromone_weight (float): Legacy positional name for beta,
          the inverse-cost influence exponent.
      local_evap_rate (float): ACS local update rate (xi).
      initial_pheromone_lvl (float | None): Baseline tau0 for the local
          update, or ``None`` when the caller has no baseline value.

  Returns:
      tuple[list, float]: The selected path and its actual aligned-edge cost.
          A lost ant ends with an infinite cost.
  """

  alpha = heuristic_weight
  beta = pheromone_weight
  tau0 = initial_pheromone_lvl
  require_edge_types = bool(graph_map.get('buses')) or any(isinstance(node, str) and node.startswith('bus:') for node in graph_map.get('node_index', []))
  validate_graph(graph_map, require_edge_types=require_edge_types)
  if start_node not in graph_map['node_index'] or end_node not in graph_map['node_index']:
    return None, np.inf
  solution_path = [start_node]
  solution_cost = 0

  # Construct a route and update each selected edge immediately
  while solution_path[-1] != end_node:
    current_node = solution_path[-1]
    neighbors = np.array(graph_map['connections'][current_node], dtype=object)
    neighbors_weights = np.array(graph_map['weights'][current_node])
    neighbors_edge_types = np.array(graph_map.get('edge_types', {}).get(current_node, ['walk'] * len(neighbors)), dtype=object)
    neighbors_pheromones = np.array(pheromone_graph[current_node])

    filter_visited_nodes_mask = ~np.isin(neighbors, solution_path)
    neighbors = neighbors[filter_visited_nodes_mask]
    neighbors_weights = neighbors_weights[filter_visited_nodes_mask]
    neighbors_edge_types = neighbors_edge_types[filter_visited_nodes_mask]
    neighbors_pheromones = neighbors_pheromones[filter_visited_nodes_mask]

    # The ant is lost. Stop the search.
    if len(neighbors) == 0:
      solution_path.append(np.inf)
      break

    selection_weights = normalize_for_selection(neighbors_weights, neighbors_edge_types)

    # Probabilistic choice of the next node (proposed by Ant Colony System ACS)
    q = np.random.rand()
    if q <= q0:
      desirability = (neighbors_pheromones**alpha) * ((1.0 / selection_weights) ** beta)
      # Degenerate desirability falls back to the strongest heuristic edge.
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
        next_node = neighbors[next_node_index - 1]

    solution_path.append(next_node)
    # Local pheromone update during route construction
    if tau0 is not None and local_evap_rate:
      edge_index = graph_map['connections'][current_node].index(next_node)
      current_pheromone = pheromone_graph[current_node][edge_index]
      pheromone_graph[current_node][edge_index] = (1 - local_evap_rate) * current_pheromone + local_evap_rate * tau0

  # return the path and calculate the incurred costs
  if solution_path[-1] != np.inf:
    for i in range(len(solution_path) - 1):
      neighbor_selected_index = graph_map['connections'][solution_path[i]].index(solution_path[i + 1])
      solution_cost += graph_map['weights'][solution_path[i]][neighbor_selected_index]
  else:
    solution_cost = np.inf

  return solution_path, solution_cost
