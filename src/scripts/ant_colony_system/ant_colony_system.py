from time import time

import numpy as np

try:
    from configuration.algorithm_settings import settings
except ModuleNotFoundError:  # Repository-root package imports.
    from src.configuration.algorithm_settings import settings

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
    stagnation_epochs=None,
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
    stagnation_epochs : int or None
        Consecutive non-improving epochs before stopping. ``None`` uses the
        configured ACS limit; a non-positive value disables early stopping.

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
    require_edge_types = bool(graph_map.get("buses")) or any(
        isinstance(node, str) and node.startswith("bus:")
        for node in graph_map.get("node_index", [])
    )
    validate_graph(graph_map, require_edge_types=require_edge_types)
    if start_node == end_node:
        if start_node in graph_map["node_index"]:
            return [start_node], 0.0, time() - start_time, 0
        return None, np.inf, time() - start_time, 0

    if initial_pheromone_lvl is None:
        baseline_cost = deterministic_route_cost(graph_map, start_node, end_node)
        if not np.isfinite(baseline_cost):
            return None, np.inf, time() - start_time, 0
        tau0 = (
            1.0
            if baseline_cost == 0
            else 1 / (len(graph_map["node_index"]) * baseline_cost)
        )
    else:
        tau0 = initial_pheromone_lvl
    alpha = heuristic_weight
    beta = pheromone_weight
    pheromone_graph = generate_pheromone_map(graph_map, tau0)
    global_best_path = None
    global_best_cost = np.inf
    epochs = 0
    stagnant_epochs = 0
    if stagnation_epochs is None:
        stagnation_epochs = settings["acs_stagnation_epochs"]

    while epochs < max_epochs:
        improved = False
        # Construct routes and apply local pheromone updates
        for ant in range(ants_number):
            route, distance = ant_solution_ACS(
                graph_map,
                pheromone_graph,
                start_node,
                end_node,
                transition_prob,
                alpha,
                beta,
                local_evap_rate,
                tau0,
            )
            if np.isfinite(distance) and distance < global_best_cost:
                global_best_path = route.copy()
                global_best_cost = distance
                improved = True

        # Global pheromone evaporation on retained global-best edges
        # Global pheromone deposition on retained global-best edges
        if global_best_path is not None:
            deposit = 0.0 if global_best_cost == 0 else 1 / global_best_cost
            for current_node, next_node in zip(
                global_best_path, global_best_path[1:]
            ):
                edge_index = graph_map["connections"][current_node].index(next_node)
                current_pheromone = pheromone_graph[current_node][edge_index]
                pheromone_graph[current_node][edge_index] = (
                    (1 - global_evap_rate) * current_pheromone
                    + global_evap_rate * deposit
                )

        # Stop when the retained best cost has not improved for long enough
        stagnant_epochs = 0 if improved else stagnant_epochs + 1
        epochs += 1
        if stagnation_epochs > 0 and stagnant_epochs >= stagnation_epochs:
            break

    return global_best_path, global_best_cost, time() - start_time, epochs
