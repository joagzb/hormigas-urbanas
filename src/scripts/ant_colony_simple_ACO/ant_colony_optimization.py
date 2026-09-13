from collections import Counter
from time import time

import numpy as np

from ..utils.algorithm_observer import notify_stage
from ..utils.generators import deterministic_route_cost, generate_pheromone_map, validate_graph
from .ant_solution_ACO import ant_solution_ACO


def ACO(graph_map, start_node, end_node, ants_number, evaporation_rate, initial_pheromone_lvl, heuristic_weight, pheromone_weight, max_epochs: int = 500, *, epoch_callback=None):
  """Find the best route seen by the Ant Colony Optimization algorithm.

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
  evaporation_rate : float
      The pheromone evaporation rate applied after each epoch.
  initial_pheromone_lvl : float or None
      Initial pheromone level. If ``None``, tau0 is derived as ``|V| / Lgb``
      from a deterministic baseline route.
  heuristic_weight : float
      Legacy positional name for alpha, the pheromone exponent.
  pheromone_weight : float
      Legacy positional name for beta, the inverse-cost exponent.
  max_epochs : int
      Maximum number of epochs to run.
  epoch_callback : callable or None
      Optional observer called after each completed epoch with the final
      ``pheromone_update`` observation.

  Returns:
  --------
  path : list of hashable or None
      The best route seen, or ``None`` if no route is found.
  cost : float
      Cost of the best route, or ``np.inf`` when no route exists.
  total_time : float
      Execution time in seconds.
  epochs : int
      Number of completed epochs.
  """
  start_time = time()
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
    tau0 = 1.0 if baseline_cost == 0 else len(graph_map['node_index']) / baseline_cost
  else:
    tau0 = initial_pheromone_lvl
  alpha = heuristic_weight
  beta = pheromone_weight
  pheromone_graph = generate_pheromone_map(graph_map, tau0)
  routes = [None] * ants_number
  distances = np.full(ants_number, np.inf)
  best_path = None
  best_cost = np.inf
  epochs = 0
  converged_ants = 0

  while converged_ants < ants_number and epochs < max_epochs:
    # Construct routes and retain the best route
    for ant in range(ants_number):
      route, distance = ant_solution_ACO(graph_map, pheromone_graph, start_node, end_node, alpha, beta)

      routes[ant] = route
      distances[ant] = distance

      if np.isfinite(distance) and distance < best_cost:
        best_path = route.copy()
        best_cost = distance

    finite_distances = distances[np.isfinite(distances)]
    if finite_distances.size:
      iteration_best_index = int(np.nanargmin(distances))
      iteration_best_path = routes[iteration_best_index].copy()
      iteration_best_cost = distances[iteration_best_index]
    else:
      iteration_best_path = None
      iteration_best_cost = np.inf

    # Global pheromone evaporation
    for pheromones in pheromone_graph.values():
      pheromones *= 1 - evaporation_rate

    # Global pheromone deposition
    for route, distance in zip(routes, distances):
      if not np.isfinite(distance):
        continue

      deposit = 0.0 if distance == 0 else 1 / distance
      for current_node, next_node in zip(route, route[1:]):
        edge_index = graph_map['connections'][current_node].index(next_node)
        pheromone_graph[current_node][edge_index] += deposit
    # Check convergence
    if finite_distances.size:
      _, converged_ants = Counter(finite_distances).most_common(1)[0]
    epochs += 1

    notify_stage(
      epoch_callback,
      epoch=epochs,
      stage='pheromone_update',
      pheromones=pheromone_graph,
      iteration_best_path=iteration_best_path,
      iteration_best_cost=iteration_best_cost,
      global_best_path=best_path,
      global_best_cost=best_cost,
    )

  return best_path, best_cost, time() - start_time, epochs
