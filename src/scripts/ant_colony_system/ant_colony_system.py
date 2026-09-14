from time import time

import numpy as np

from ..utils.algorithm_observer import notify_stage
from ..utils.algorithm_termination import has_path_consensus, has_stable_iteration_best_cost, validate_path_consensus_threshold
from ..utils.generators import deterministic_route_cost, generate_pheromone_map, validate_graph
from .ant_solution_ACS import ant_solution_ACS


def ACS(
  graph_map,
  start_node,
  end_node,
  ants_number,
  global_evap_rate,
  local_evap_rate,
  transition_prob,
  initial_pheromone_lvl,
  heuristic_weight,
  pheromone_weight,
  max_epochs: int = 500,
  *,
  path_consensus_threshold=0.85,
  stagnation_epochs=None,
  epoch_callback=None,
):
  """Find and retain the global-best route using Ant Colony System.

  Parameters:
  -----------
  graph_map : dict
      Graph containing aligned ``connections`` and ``weights`` mappings and a
      ``node_index`` collection.
  start_node : hashable
      The opaque starting node ID (ant hill).
  end_node : hashable
      The opaque destination node ID (food).
  ants_number : int
      The number of ants used in each epoch.
  global_evap_rate : float
      Evaporation rate used for the global-best pheromone update.
  local_evap_rate : float
      Evaporation rate applied as each edge is selected.
  transition_prob : float
      Probability of selecting the strongest transition instead of roulette
      selection.
  initial_pheromone_lvl : float or None
      Initial pheromone level. If ``None``, tau0 is derived as
      ``1 / (|V| * Lgb)`` from a deterministic baseline route.
  heuristic_weight : float
      Legacy positional name for alpha, the pheromone exponent.
  pheromone_weight : float
      Legacy positional name for beta, the inverse-cost exponent.
  max_epochs : int
      Maximum number of epochs to run.
  path_consensus_threshold : float
      Fraction of finite ants that must complete the same route before stable
      consecutive iteration-best costs can stop the search.
  stagnation_epochs : int or None
      Deprecated compatibility argument. It is ignored and does not affect
      consensus termination.
  epoch_callback : callable or None
      Optional observer called after each completed epoch with the final
      ``pheromone_update`` observation.

  Returns:
  --------
  path : list of hashable or None
      The retained global-best route, or ``None`` if no route exists.
  cost : float
      Cost of the retained route, or ``np.inf`` when no route exists.
  total_time : float
      Execution time in seconds.
  epochs : int
      Number of completed epochs.
  """
  start_time = time()
  validate_path_consensus_threshold(path_consensus_threshold)
  require_edge_types = bool(graph_map.get('buses')) or any(isinstance(node, str) and node.startswith('bus:') for node in graph_map.get('node_index', []))
  validate_graph(graph_map, require_edge_types=require_edge_types)
  if start_node == end_node:
    if start_node in graph_map['node_index']:
      return [start_node], 0.0, time() - start_time, 0
    return None, np.inf, time() - start_time, 0

  if initial_pheromone_lvl is None:
    baseline_cost = deterministic_route_cost(graph_map, start_node, end_node)
    if not np.isfinite(baseline_cost):
      return None, np.inf, time() - start_time, 0
    tau0 = 1.0 if baseline_cost == 0 else 1 / (len(graph_map['node_index']) * baseline_cost)
  else:
    tau0 = initial_pheromone_lvl
  alpha = heuristic_weight
  beta = pheromone_weight
  pheromone_graph = generate_pheromone_map(graph_map, tau0)
  global_best_path = None
  global_best_cost = np.inf
  epochs = 0
  previous_iteration_best_cost = None

  while epochs < max_epochs:
    routes = [None] * ants_number
    distances = np.full(ants_number, np.inf)
    iteration_best_path = None
    iteration_best_cost = np.inf
    # Construct routes and apply local pheromone updates
    for ant in range(ants_number):
      route, distance = ant_solution_ACS(graph_map, pheromone_graph, start_node, end_node, transition_prob, alpha, beta, local_evap_rate, tau0)
      routes[ant] = route
      distances[ant] = distance
      if np.isfinite(distance) and distance < iteration_best_cost:
        iteration_best_path = route.copy()
        iteration_best_cost = distance

      if np.isfinite(distance) and distance < global_best_cost:
        global_best_path = route.copy()
        global_best_cost = distance

    # Global pheromone evaporation on retained global-best edges
    # Global pheromone deposition on retained global-best edges
    if global_best_path is not None:
      deposit = 0.0 if global_best_cost == 0 else 1 / global_best_cost
      for current_node, next_node in zip(global_best_path, global_best_path[1:]):
        edge_index = graph_map['connections'][current_node].index(next_node)
        current_pheromone = pheromone_graph[current_node][edge_index]
        pheromone_graph[current_node][edge_index] = (1 - global_evap_rate) * current_pheromone + global_evap_rate * deposit
    epochs += 1

    notify_stage(
      epoch_callback,
      epoch=epochs,
      stage='pheromone_update',
      pheromones=pheromone_graph,
      iteration_best_path=iteration_best_path,
      iteration_best_cost=iteration_best_cost,
      global_best_path=global_best_path,
      global_best_cost=global_best_cost,
    )

    consensus_reached = has_path_consensus(routes, distances, path_consensus_threshold)
    stable_iteration_best = has_stable_iteration_best_cost(previous_iteration_best_cost, iteration_best_cost)
    previous_iteration_best_cost = iteration_best_cost
    if consensus_reached and stable_iteration_best:
      break

  return global_best_path, global_best_cost, time() - start_time, epochs
