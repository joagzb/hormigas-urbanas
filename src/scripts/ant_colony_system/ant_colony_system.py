from time import time

import numpy as np

from ..utils.algorithm_observer import record_stage_data
from ..utils.algorithm_validations import update_global_best, validate_global_best
from ..utils.generators import deterministic_route_cost, generate_pheromone_map
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
  global_best_patience=10,
  epoch_callback=None,
):
  """Find and retain the global-best route using Ant Colony System.

  Parameters:
  -----------
  graph_map : dict
      Preflighted graph containing opaque IDs and aligned ``connections``,
      ``weights``, and ``edge_types`` mappings.

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
      Consecutive completed epochs without strict global-best improvement
      before stopping.

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

  # initial settings
  validate_global_best(global_best_patience)

  if initial_pheromone_lvl is None:
    baseline_cost = deterministic_route_cost(graph_map, start_node, end_node)
    if not np.isfinite(baseline_cost):
      return None, np.inf, time() - start_time, 0
    if baseline_cost == 0:
      initial_pheromone_lvl = 1.0
    else:
      initial_pheromone_lvl = 1 / (len(graph_map['node_index']) * baseline_cost)

  alpha = heuristic_weight
  beta = pheromone_weight
  pheromone_graph = generate_pheromone_map(graph_map, initial_pheromone_lvl)
  global_best_path = None
  global_best_cost = np.inf
  epochs = 0
  epochs_without_global_best_improvement = 0

  while epochs < max_epochs:
    routes = [None] * ants_number
    distances = np.full(ants_number, np.inf)
    iteration_best_path = None
    iteration_best_cost = np.inf

    # each ant constructs a route
    for ant in range(ants_number):
      route, distance = ant_solution_ACS(graph_map, pheromone_graph, start_node, end_node, transition_prob, alpha, beta, local_evap_rate, initial_pheromone_lvl)

      routes[ant] = route
      distances[ant] = distance

      # retain the best route
      if np.isfinite(distance) and distance < iteration_best_cost:
        iteration_best_path = route.copy()
        iteration_best_cost = distance

    improved, epochs_without_global_best_improvement = update_global_best(global_best_cost, iteration_best_cost, epochs_without_global_best_improvement)
    if improved:
      global_best_path = iteration_best_path.copy()
      global_best_cost = iteration_best_cost

    # Global pheromone evaporation and deposition on retained global-best edges
    if global_best_path is not None:
      if global_best_cost == 0:
        deposit = 0.0
      else:
        deposit = 1 / global_best_cost
      for current_node, next_node in zip(global_best_path, global_best_path[1:]):
        edge_index = graph_map['connections'][current_node].index(next_node)
        current_pheromone = pheromone_graph[current_node][edge_index]
        pheromone_graph[current_node][edge_index] = (1 - global_evap_rate) * current_pheromone + global_evap_rate * deposit

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
      global_best_path=global_best_path,
      global_best_cost=global_best_cost,
    )

    if epochs_without_global_best_improvement >= validate_global_best(global_best_patience):
      break

  return global_best_path, global_best_cost, time() - start_time, epochs
