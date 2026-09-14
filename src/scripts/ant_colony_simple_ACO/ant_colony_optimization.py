from time import time

import numpy as np

from ..utils.algorithm_observer import record_stage_data
from ..utils.algorithm_validations import update_global_best, validate_global_best
from ..utils.generators import deterministic_route_cost, generate_pheromone_map
from .ant_solution_ACO import ant_solution_ACO


def ACO(
  graph_map,
  start_node,
  end_node,
  ants_number,
  evaporation_rate,
  initial_pheromone_lvl,
  heuristic_weight,
  pheromone_weight,
  max_epochs: int = 500,
  *,
  global_best_patience=10,
  epoch_callback=None,
):
  """Find the best route by the Ant Colony Optimization algorithm.

  Parameters:
  -----------
  graph_map : dict
      graph representing a city with opaque IDs and aligned ``connections``,
      ``weights``, and ``edge_types`` mappings.

  start_node : hashable
      The opaque starting node ID (ant hill).

  end_node : hashable
      The opaque destination node ID (food).

  ants_number : int
      The number of ants used in each epoch.

  evaporation_rate : float
      The pheromone evaporation rate applied after each epoch.

  initial_pheromone_lvl : float or None
      Initial pheromone level. If ``None``, an automatic baseline is derived
      from a deterministic reference route. Consult the algorithm
      documentation for the theoretical initialization formula.

  heuristic_weight : float
      Legacy positional name for alpha, the pheromone exponent.

  pheromone_weight : float
      Legacy positional name for beta, the inverse-cost exponent.

  max_epochs : int
      Maximum number of epochs to run.

  global_best_patience : int
      Consecutive completed epochs without strict finite global-best
      improvement before stopping.

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

  # initial settings
  validate_global_best(global_best_patience)

  if initial_pheromone_lvl is None:
    baseline_cost = deterministic_route_cost(graph_map, start_node, end_node)
    if not np.isfinite(baseline_cost):
      return None, np.inf, time() - start_time, 0
    if baseline_cost == 0:
      initial_pheromone_lvl = 1.0
    else:
      initial_pheromone_lvl = len(graph_map['node_index']) / baseline_cost

  alpha = heuristic_weight
  beta = pheromone_weight
  pheromone_graph = generate_pheromone_map(graph_map, initial_pheromone_lvl)
  routes = [None] * ants_number
  distances = np.full(ants_number, np.inf)
  best_path = None
  best_cost = np.inf
  epochs = 0
  epochs_without_global_best_improvement = 0

  while epochs < max_epochs:
    # each ant constructs a route
    for ant in range(ants_number):
      route, distance = ant_solution_ACO(graph_map, pheromone_graph, start_node, end_node, alpha, beta)

      routes[ant] = route
      distances[ant] = distance

    # retain the best route
    finite_distances = distances[np.isfinite(distances)]
    if finite_distances.size:
      iteration_best_index = int(np.nanargmin(distances))
      iteration_best_path = routes[iteration_best_index].copy()
      iteration_best_cost = distances[iteration_best_index]
    else:
      iteration_best_path = None
      iteration_best_cost = np.inf

    improved, epochs_without_global_best_improvement = update_global_best(best_cost, iteration_best_cost, epochs_without_global_best_improvement)
    if improved:
      best_path = iteration_best_path.copy()
      best_cost = iteration_best_cost

    # Global pheromone evaporation
    for pheromones in pheromone_graph.values():
      pheromones *= 1 - evaporation_rate

    # Global pheromone deposition
    for route, distance in zip(routes, distances):
      if not np.isfinite(distance):
        continue

      if distance == 0:
        deposit = 0.0
      else:
        deposit = 1 / distance
      for current_node, next_node in zip(route, route[1:]):
        edge_index = graph_map['connections'][current_node].index(next_node)
        pheromone_graph[current_node][edge_index] += deposit

    # advance to next epoch
    epochs += 1

    # record epoch information
    record_stage_data(
      epoch_callback,
      epoch=epochs,
      stage='pheromone_update',
      pheromones=pheromone_graph,
      iteration_best_path=iteration_best_path,
      iteration_best_cost=iteration_best_cost,
      global_best_path=best_path,
      global_best_cost=best_cost,
    )

    # TODO: CAN I PUT WHAT IS IN THE LINE 112 ABOVE THIS CODE?
    if epochs_without_global_best_improvement >= validate_global_best(global_best_patience):
      break

  return best_path, best_cost, time() - start_time, epochs
