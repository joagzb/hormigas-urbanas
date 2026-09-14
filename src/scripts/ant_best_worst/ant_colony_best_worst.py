from time import time

import numpy as np

try:
  from configuration.algorithm_settings import settings
except ModuleNotFoundError:  # Repository-root package imports.
  from src.configuration.algorithm_settings import settings

from ..utils.algorithm_observer import record_stage_data
from ..utils.algorithm_validations import update_global_best, validate_global_best
from ..utils.generators import deterministic_route_cost, generate_pheromone_map
from .ant_solution_ABW import ant_solution_best_worst


def _mutate_pheromone_rows(pheromone_graph, mutation_probability, mutation_scale, current_epoch, last_restart_epoch, max_epochs, global_best_mean, min_pheromone_lvl):
  search_progress = (current_epoch - last_restart_epoch) / max(1, max_epochs)
  mutation_amount = mutation_scale * max(0.0, search_progress) * global_best_mean

  for pheromones in pheromone_graph.values():
    if np.random.random() >= mutation_probability:
      continue
    if np.random.randint(2):
      direction = 1
    else:
      direction = -1
    pheromones += direction * mutation_amount
    pheromones[pheromones < min_pheromone_lvl] = min_pheromone_lvl


def ABW(
  graph_map,
  start_node,
  end_node,
  ants_number,
  global_evap_rate,
  max_epochs,
  initial_pheromone_lvl,
  heuristic_weight,
  pheromone_weight,
  worst_penalty_rate=None,
  mutation_probability=0.05,
  mutation_scale=2.0,
  restart_stagnation=None,
  min_pheromone_lvl=None,
  global_best_patience=10,
  epoch_callback=None,
):
  """Find the global-best route using Best-Worst Ant System.

  Parameters:
  -----------
  - graph_map : dict
      Preflighted graph containing opaque IDs and aligned ``connections``,
      ``weights``, and ``edge_types`` mappings.

  - start_node : hashable
      The opaque starting node ID (ant hill).

  - end_node : hashable
      The opaque destination node ID (food).

  - ants_number : int
      The number of ants used in each epoch.

  - global_evap_rate : float
      Pheromone evaporation rate applied after each epoch.

  - max_epochs : int
      Maximum number of epochs to run.

  - initial_pheromone_lvl : float or None
      Initial pheromone level. If ``None``, an automatic baseline is derived
      from a deterministic reference route. Consult the algorithm
      documentation for the theoretical initialization formula.

  - heuristic_weight : float
      Legacy positional name for alpha, the pheromone exponent.

  - pheromone_weight : float
      Legacy positional name for beta, the inverse-cost exponent.

  - worst_penalty_rate : float or None
      Extra evaporation rate for worst-route edges. Defaults to
      ``global_evap_rate``.

  - mutation_probability : float
      Probability of mutating each pheromone row per epoch.

  - mutation_scale : float
      Scale applied to search-progress-based mutation.

  - restart_stagnation : int or None
      Consecutive non-improving epochs before pheromone levels restart. A
      ``None`` value uses the configured BWAS restart limit; a non-positive
      value disables restarts.

  - min_pheromone_lvl : float or None
      Positive pheromone floor. A ``None`` value uses the base configuration
      for backward compatibility.

  - global_best_patience : int
      Consecutive completed epochs without strict global-best improvement
      before stopping.

  - epoch_callback : callable or None
      Optional observer called after each completed epoch and any restart,
      with the final ``pheromone_update`` observation.

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

  if worst_penalty_rate is None:
    penalty_rate = global_evap_rate
  else:
    penalty_rate = worst_penalty_rate

  if min_pheromone_lvl is None:
    min_pheromone_lvl = settings['f_min']
  # The configured floor is an invariant, including initial and restarted trails.
  restart_pheromone_lvl = max(initial_pheromone_lvl, min_pheromone_lvl)
  pheromone_graph = generate_pheromone_map(graph_map, restart_pheromone_lvl)
  routes = [None] * ants_number
  distances = np.full(ants_number, np.inf)
  global_best_path = None
  global_best_cost = np.inf
  restart_stagnant_epochs = 0
  if restart_stagnation is None:
    restart_stagnation = settings['bwas_restart_stagnation']
  last_restart_epoch = 0
  epochs = 0
  epochs_without_global_best_improvement = 0

  while epochs < max_epochs:
    for ant in range(ants_number):
      # each ant constructs a route
      route, distance = ant_solution_best_worst(graph_map, pheromone_graph, start_node, end_node, alpha, beta)
      routes[ant] = route
      distances[ant] = distance

    # Select the best finite route for this iteration and the finite worst route.
    iteration_best_path = None
    iteration_best_cost = np.inf
    finite_indices = np.flatnonzero(np.isfinite(distances))
    worst_route = None
    if finite_indices.size:
      best_index = finite_indices[np.argmin(distances[finite_indices])]
      worst_index = finite_indices[np.argmax(distances[finite_indices])]
      iteration_best_cost = distances[best_index]
      iteration_best_path = routes[best_index].copy()
      worst_route = routes[worst_index]

    improved, epochs_without_global_best_improvement = update_global_best(global_best_cost, iteration_best_cost, epochs_without_global_best_improvement)
    if improved:
      global_best_path = iteration_best_path.copy()
      global_best_cost = iteration_best_cost

    # Global pheromone evaporation
    for pheromones in pheromone_graph.values():
      pheromones *= 1 - global_evap_rate

    # Global pheromone deposition on retained global-best edges
    global_best_edges = set()
    if global_best_path is not None:
      if global_best_cost == 0:
        deposit = 0.0
      else:
        deposit = 1 / global_best_cost
      for current_node, next_node in zip(global_best_path, global_best_path[1:]):
        global_best_edges.add((current_node, next_node))
        edge_index = graph_map['connections'][current_node].index(next_node)
        pheromone_graph[current_node][edge_index] += deposit

    # Penalize worst-path found by ant
    if worst_route is not None:
      for current_node, next_node in zip(worst_route, worst_route[1:]):
        if (current_node, next_node) in global_best_edges:
          continue
        edge_index = graph_map['connections'][current_node].index(next_node)
        pheromone_graph[current_node][edge_index] *= 1 - penalty_rate

    # Mutate pheromones
    if global_best_path is not None:
      best_pheromones = []
      for current_node, next_node in zip(global_best_path, global_best_path[1:]):
        edge_index = graph_map['connections'][current_node].index(next_node)
        best_pheromones.append(pheromone_graph[current_node][edge_index])
      global_best_mean = float(np.mean(best_pheromones))
      _mutate_pheromone_rows(pheromone_graph, mutation_probability, mutation_scale, epochs, last_restart_epoch, max_epochs, global_best_mean, min_pheromone_lvl)

    for pheromones in pheromone_graph.values():
      pheromones[pheromones < min_pheromone_lvl] = min_pheromone_lvl

    # Restart pheromones to restore trail diversity while retaining the global best
    if improved:
      restart_stagnant_epochs = 0
    else:
      restart_stagnant_epochs += 1
    restarted = False
    if restart_stagnation > 0 and restart_stagnant_epochs >= restart_stagnation:
      pheromone_graph = generate_pheromone_map(graph_map, restart_pheromone_lvl)
      last_restart_epoch = epochs
      restart_stagnant_epochs = 0
      restarted = True

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
      restarted=restarted,
    )

    if epochs_without_global_best_improvement >= validate_global_best(global_best_patience):
      break

  return global_best_path, global_best_cost, time() - start_time, epochs
