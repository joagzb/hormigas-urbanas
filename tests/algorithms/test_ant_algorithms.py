import importlib
import inspect
import warnings

import numpy as np
import pytest

from src.scripts.ant_colony_simple_ACO.ant_solution_ACO import ant_solution_ACO
from src.scripts.ant_colony_system.ant_solution_ACS import ant_solution_ACS
from src.scripts.ant_best_worst.ant_solution_ABW import ant_solution_best_worst
from src.scripts.utils.generators import merge_bus_and_map_graph
from src.scripts.utils.toy_city_generators import (
    generate_bus_line_square_city,
    generate_square_city_graph,
)


GRAPH = {
    "node_index": {0, 1, 2, 3},
    "connections": {0: [1, 2], 1: [3], 2: [3], 3: []},
    "weights": {0: [1.0, 4.0], 1: [1.0], 2: [1.0], 3: []},
    "edge_types": {0: ["walk", "walk"], 1: ["walk"], 2: ["walk"], 3: []},
}

UNREACHABLE_GRAPH = {
    "node_index": {0, 1, 2},
    "connections": {0: [1], 1: [], 2: []},
    "weights": {0: [1.0], 1: [], 2: []},
}

EMPTY_ADJACENCY_GRAPH = {
    "node_index": {0, 1},
    "connections": {0: [], 1: []},
    "weights": {0: [], 1: []},
}

START_NODE = 0
END_NODE = 3
ANTS_NUMBER = 3
EVAPORATION_RATE = 0.1
LOCAL_EVAPORATION_RATE = 0.1
TRANSITION_PROBABILITY = 0.8
INITIAL_PHEROMONE_LVL = None
HEURISTIC_WEIGHT = 1
PHEROMONE_WEIGHT = 1
MAX_EPOCHS = 3


def _capture_generated_pheromones(captured):
    def capture_generator(graph, level):
        captured.update(
            {
                node: np.full(len(edges), level)
                for node, edges in graph["connections"].items()
            }
        )
        return captured

    return capture_generator


def _run_aco(graph, start_node, end_node):
    module = importlib.import_module(
        "src.scripts.ant_colony_simple_ACO.ant_colony_optimization"
    )
    return module.ACO(
        graph,
        start_node,
        end_node,
        ANTS_NUMBER,
        EVAPORATION_RATE,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        MAX_EPOCHS,
    )


def _run_acs(graph, start_node, end_node):
    module = importlib.import_module("src.scripts.ant_colony_system.ant_colony_system")
    return module.ACS(
        graph,
        start_node,
        end_node,
        ANTS_NUMBER,
        EVAPORATION_RATE,
        LOCAL_EVAPORATION_RATE,
        TRANSITION_PROBABILITY,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        MAX_EPOCHS,
    )


def _run_bwas(graph, start_node, end_node):
    module = importlib.import_module(
        "src.scripts.ant_best_worst.ant_colony_best_worst"
    )
    return module.ABW(
        graph,
        start_node,
        end_node,
        ANTS_NUMBER,
        EVAPORATION_RATE,
        MAX_EPOCHS,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
    )


ALGORITHM_RUNNERS = [_run_aco, _run_acs, _run_bwas]


def _multimodal_square_graph():
    map_graph = generate_square_city_graph(2, fixed_weight=10.0)
    bus_services = generate_bus_line_square_city(
        2,
        fixed_weight=1.0,
        line_id="E2E",
        route=[0, 2],
    )
    return merge_bus_and_map_graph(map_graph, bus_services)


@pytest.mark.parametrize("run_algorithm", ALGORITHM_RUNNERS)
def test_derived_pheromone_returns_trivial_route_when_start_equals_end(run_algorithm):
    path, cost, _, epochs = run_algorithm(EMPTY_ADJACENCY_GRAPH, 0, 0)

    assert (path, cost, epochs) == ([0], 0.0, 0)


@pytest.mark.parametrize("run_algorithm", ALGORITHM_RUNNERS)
def test_trivial_route_requires_existing_node(run_algorithm):
    path, cost, _, epochs = run_algorithm(EMPTY_ADJACENCY_GRAPH, 99, 99)

    assert path is None
    assert np.isinf(cost)
    assert epochs == 0


@pytest.mark.parametrize("run_algorithm", ALGORITHM_RUNNERS)
@pytest.mark.parametrize(
    ("graph", "end_node"),
    [(UNREACHABLE_GRAPH, 2), (EMPTY_ADJACENCY_GRAPH, 1)],
)
def test_derived_pheromone_returns_no_route_for_unreachable_graph(
    run_algorithm, graph, end_node
):
    path, cost, _, epochs = run_algorithm(graph, 0, end_node)

    assert path is None
    assert np.isinf(cost)
    assert epochs == 0


@pytest.mark.parametrize(
    ("run_algorithm", "module_name", "ant_name", "expected_tau0"),
    [
        (
            _run_aco,
            "src.scripts.ant_colony_simple_ACO.ant_colony_optimization",
            "ant_solution_ACO",
            2.0,
        ),
        (
            _run_acs,
            "src.scripts.ant_colony_system.ant_colony_system",
            "ant_solution_ACS",
            0.125,
        ),
        (
            _run_bwas,
            "src.scripts.ant_best_worst.ant_colony_best_worst",
            "ant_solution_best_worst",
            0.125,
        ),
    ],
)
def test_derived_pheromone_initialization_does_not_invoke_dijkstra(
    monkeypatch, run_algorithm, module_name, ant_name, expected_tau0
):
    generators = importlib.import_module("src.scripts.utils.generators")
    route_finder = importlib.import_module("src.scripts.utils.route_finder")
    module = importlib.import_module(module_name)
    generated_levels = []
    original_generator = module.generate_pheromone_map

    def fail_if_called(*args):
        raise AssertionError("Dijkstra must remain an external evaluation oracle")

    def capture_generator(graph, level):
        generated_levels.append(level)
        return original_generator(graph, level)

    monkeypatch.setattr(route_finder, "dijkstra", fail_if_called)
    monkeypatch.setattr(module, "generate_pheromone_map", capture_generator)
    monkeypatch.setattr(module, ant_name, lambda *args: ([0, 1, 3], 2.0))

    path, cost, _, _ = run_algorithm(GRAPH, START_NODE, END_NODE)

    assert "dijkstra" not in inspect.getsource(generators)
    assert "dijkstra" not in inspect.getsource(module)
    assert generated_levels[0] == pytest.approx(expected_tau0)
    assert (path, cost) == ([0, 1, 3], 2.0)


@pytest.mark.parametrize("ant_solution", [ant_solution_ACO, ant_solution_best_worst])
def test_aco_and_bwas_transitions_use_actual_aligned_edge_cost(
    monkeypatch, ant_solution
):
    observed = {}

    def select_first(probabilities):
        observed.setdefault("probabilities", probabilities)
        return 1

    module = importlib.import_module(ant_solution.__module__)
    monkeypatch.setattr(module, "roulette_wheel_selection", select_first)
    pheromones = {
        0: np.array([1.0, 1.0]),
        1: np.array([1.0]),
        2: np.array([1.0]),
        3: np.array([]),
    }

    ant_solution(
        GRAPH,
        pheromones,
        START_NODE,
        END_NODE,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
    )

    assert observed["probabilities"] == pytest.approx([0.8, 0.2])


@pytest.mark.parametrize(
    "ant_solution", [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst]
)
@pytest.mark.parametrize(("start_node", "end_node"), [(99, 99), (99, 1), (0, 99)])
def test_direct_ant_solutions_reject_nonexistent_endpoints(
    ant_solution, start_node, end_node
):
    pheromones = {0: np.array([]), 1: np.array([])}
    arguments = [EMPTY_ADJACENCY_GRAPH, pheromones, start_node, end_node]
    if ant_solution is ant_solution_ACS:
        arguments.append(TRANSITION_PROBABILITY)
    arguments.extend([HEURISTIC_WEIGHT, PHEROMONE_WEIGHT])

    path, cost = ant_solution(*arguments)

    assert path is None
    assert np.isinf(cost)


@pytest.mark.parametrize(
    "ant_solution", [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst]
)
def test_direct_ant_solutions_reject_multimodal_graph_without_edge_types(ant_solution):
    graph = _multimodal_square_graph()
    graph.pop("edge_types")
    pheromones = {
        node: np.ones(len(connections))
        for node, connections in graph["connections"].items()
    }
    arguments = [graph, pheromones, 0, 2]
    if ant_solution is ant_solution_ACS:
        arguments.append(TRANSITION_PROBABILITY)
    arguments.extend([HEURISTIC_WEIGHT, PHEROMONE_WEIGHT])

    with pytest.raises(ValueError, match="edge_types"):
        ant_solution(*arguments)


@pytest.mark.parametrize(
    "ant_solution", [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst]
)
def test_direct_ant_solutions_reject_tagged_bus_nodes_with_empty_buses(ant_solution):
    bus_node = "bus:line:outbound:0"
    graph = {
        "node_index": {0, bus_node, 1},
        "connections": {0: [bus_node], bus_node: [1], 1: []},
        "weights": {0: [1.4], bus_node: [0.3], 1: []},
        "buses": [],
    }
    pheromones = {0: np.array([1.0]), bus_node: np.array([1.0]), 1: np.array([])}
    arguments = [graph, pheromones, 0, 1]
    if ant_solution is ant_solution_ACS:
        arguments.append(TRANSITION_PROBABILITY)
    arguments.extend([HEURISTIC_WEIGHT, PHEROMONE_WEIGHT])

    with pytest.raises(ValueError, match="edge_types"):
        ant_solution(*arguments)


@pytest.mark.parametrize(
    "ant_solution", [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst]
)
@pytest.mark.parametrize("metadata_problem", ["missing_node", "misaligned_row"])
def test_direct_ant_solutions_reject_incomplete_or_malformed_edge_types(
    ant_solution, metadata_problem
):
    graph = _multimodal_square_graph()
    if metadata_problem == "missing_node":
        graph["edge_types"].pop(next(iter(graph["node_index"])))
    else:
        graph["edge_types"][0] = []
    pheromones = {
        node: np.ones(len(connections))
        for node, connections in graph["connections"].items()
    }
    arguments = [graph, pheromones, 0, 2]
    if ant_solution is ant_solution_ACS:
        arguments.append(TRANSITION_PROBABILITY)
    arguments.extend([HEURISTIC_WEIGHT, PHEROMONE_WEIGHT])

    with pytest.raises(ValueError, match="edge_types"):
        ant_solution(*arguments)


@pytest.mark.parametrize(
    "ant_solution", [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst]
)
def test_direct_ant_solutions_keep_plain_walking_graph_compatibility(ant_solution):
    graph = {
        "node_index": {0, 1},
        "connections": {0: [1], 1: []},
        "weights": {0: [1.0], 1: []},
    }
    pheromones = {0: np.array([1.0]), 1: np.array([])}
    arguments = [graph, pheromones, 0, 1]
    if ant_solution is ant_solution_ACS:
        arguments.append(TRANSITION_PROBABILITY)
    arguments.extend([HEURISTIC_WEIGHT, PHEROMONE_WEIGHT])

    path, cost = ant_solution(*arguments)

    assert path == [0, 1]
    assert cost == pytest.approx(1.0)


@pytest.mark.parametrize(
    "ant_solution", [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst]
)
def test_transitions_use_normalized_weights_but_return_original_cost(
    monkeypatch, ant_solution
):
    graph = {
        "node_index": {0, 1, 2},
        "connections": {0: [1, 2], 1: [], 2: []},
        "weights": {0: [0.3, 0.01], 1: [], 2: []},
        "edge_types": {0: ["ride", "alight"], 1: [], 2: []},
    }
    pheromones = {0: np.array([1.0, 1.0]), 1: np.array([]), 2: np.array([])}
    module = importlib.import_module(ant_solution.__module__)
    monkeypatch.setattr(
        module,
        "roulette_wheel_selection",
        lambda probabilities: int(np.argmax(probabilities)) + 1,
    )

    arguments = [graph, pheromones, 0, 1]
    if ant_solution is ant_solution_ACS:
        arguments.append(1.0)
    arguments.extend([1.0, 2.0])
    path, cost = ant_solution(*arguments)

    assert path == [0, 1]
    assert cost == pytest.approx(0.3)


@pytest.mark.parametrize("ant_solution", [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst])
def test_transitions_preserve_opaque_bus_node_ids(monkeypatch, ant_solution):
    bus_node = "bus:line:outbound:0"
    graph = {
        "node_index": {0, bus_node, 1},
        "connections": {0: [bus_node], bus_node: [1], 1: []},
        "weights": {0: [1.4], bus_node: [0.3], 1: []},
        "edge_types": {0: ["board"], bus_node: ["ride"], 1: []},
    }
    pheromones = {0: np.array([1.0]), bus_node: np.array([1.0]), 1: np.array([])}
    module = importlib.import_module(ant_solution.__module__)
    monkeypatch.setattr(module, "roulette_wheel_selection", lambda probabilities: 1)

    arguments = [graph, pheromones, 0, 1]
    if ant_solution is ant_solution_ACS:
        arguments.append(0.0)
    arguments.extend([1.0, 1.0])
    path, cost = ant_solution(*arguments)

    assert path == [0, bus_node, 1]
    assert cost == pytest.approx(1.7)


def test_aco_derives_tau0_and_retains_best_so_far(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_colony_simple_ACO.ant_colony_optimization"
    )
    generated_levels = []
    original_generator = module.generate_pheromone_map

    def capture_generator(graph, level):
        generated_levels.append(level)
        return original_generator(graph, level)

    solutions = iter(
        [
            ([0, 1, 3], 2.0),
            ([0, 2, 3], 5.0),
            ([0, 2, 3], 5.0),
            ([0, 1, 3], 4.0),
        ]
    )
    monkeypatch.setattr(module, "generate_pheromone_map", capture_generator)
    monkeypatch.setattr(module, "ant_solution_ACO", lambda *args: next(solutions))

    path, cost, _, epochs = module.ACO(
        GRAPH,
        START_NODE,
        END_NODE,
        2,
        EVAPORATION_RATE,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        2,
    )

    assert generated_levels == [2.0]  # |V| / Lgb = 4 / 2
    assert (path, cost, epochs) == ([0, 1, 3], 2.0, 2)


def test_acs_updates_selected_edge_immediately_toward_tau0(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_colony_system.ant_solution_ACS"
    )
    monkeypatch.setattr(module.np.random, "rand", lambda: 0.0)
    pheromones = {
        0: np.array([1.0, 1.0]),
        1: np.array([1.0]),
        2: np.array([1.0]),
        3: np.array([]),
    }

    path, _ = ant_solution_ACS(
        GRAPH,
        pheromones,
        START_NODE,
        END_NODE,
        1.0,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        0.5,
        0.2,
    )

    assert path == [0, 1, 3]
    assert pheromones[0][0] == pytest.approx(0.6)
    assert pheromones[1][0] == pytest.approx(0.6)
    assert pheromones[0][1] == pytest.approx(1.0)


def test_acs_global_update_only_touches_retained_global_best(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_colony_system.ant_colony_system"
    )
    captured = {}

    solutions = iter([([0, 1, 3], 2.0), ([0, 2, 3], 5.0)])
    monkeypatch.setattr(
        module, "generate_pheromone_map", _capture_generated_pheromones(captured)
    )
    monkeypatch.setattr(module, "ant_solution_ACS", lambda *args: next(solutions))

    path, cost, _, _ = module.ACS(
        GRAPH,
        START_NODE,
        END_NODE,
        2,
        0.5,
        LOCAL_EVAPORATION_RATE,
        1,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        1,
    )

    assert (path, cost) == ([0, 1, 3], 2.0)
    assert captured[0][0] == pytest.approx(0.3125)
    assert captured[1][0] == pytest.approx(0.3125)
    assert captured[0][1] == pytest.approx(0.125)
    assert captured[2][0] == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("module_name", "colony_name", "ant_name", "arguments", "kwargs"),
    [
        (
            "src.scripts.ant_colony_simple_ACO.ant_colony_optimization",
            "ACO",
            "ant_solution_ACO",
            (0, 1, 1, 0.1, 1.0, 1, 1, 1),
            {},
        ),
        (
            "src.scripts.ant_colony_system.ant_colony_system",
            "ACS",
            "ant_solution_ACS",
            (0, 1, 1, 0.1, 0.1, 1.0, 1.0, 1, 1, 1),
            {},
        ),
        (
            "src.scripts.ant_best_worst.ant_colony_best_worst",
            "ABW",
            "ant_solution_best_worst",
            (0, 1, 1, 0.1, 1, 1.0, 1, 1),
            {"mutation_probability": 0},
        ),
    ],
)
def test_zero_cost_baseline_uses_finite_tau0_and_zero_deposit(
    monkeypatch, module_name, colony_name, ant_name, arguments, kwargs
):
    graph = {
        "node_index": {0, 1},
        "connections": {0: [1], 1: []},
        "weights": {0: [0.0], 1: []},
    }
    module = importlib.import_module(module_name)
    captured = {}
    generated_levels = []

    def capture_generator(graph_map, level):
        generated_levels.append(level)
        return _capture_generated_pheromones(captured)(graph_map, level)

    monkeypatch.setattr(module, "generate_pheromone_map", capture_generator)
    monkeypatch.setattr(module, ant_name, lambda *args: ([0, 1], 0.0))

    arguments = list(arguments)
    initial_pheromone_index = {"ACO": 4, "ACS": 6, "ABW": 5}[colony_name]
    arguments[initial_pheromone_index] = None

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        path, cost, _, _ = getattr(module, colony_name)(graph, *arguments, **kwargs)

    assert (path, cost) == ([0, 1], 0.0)
    assert generated_levels == [1.0]
    assert captured[0] == pytest.approx([0.9])
    assert np.isfinite(captured[0]).all()


def test_acs_returns_best_route_seen_across_epochs(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_colony_system.ant_colony_system"
    )
    solutions = iter(
        [
            ([0, 1, 3], 2.0),
            ([0, 2, 3], 5.0),
            ([0, 2, 3], 5.0),
            ([0, 1, 3], 4.0),
        ]
    )
    monkeypatch.setattr(module, "ant_solution_ACS", lambda *args: next(solutions))

    path, cost, _, epochs = module.ACS(
        GRAPH,
        START_NODE,
        END_NODE,
        2,
        EVAPORATION_RATE,
        LOCAL_EVAPORATION_RATE,
        1,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        2,
    )

    assert (path, cost, epochs) == ([0, 1, 3], 2.0, 2)


def test_acs_stops_after_configured_non_improving_epochs(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_colony_system.ant_colony_system"
    )
    solutions = iter(
        [([0, 1, 3], 2.0)] + [([0, 2, 3], 5.0)] * 3
    )
    monkeypatch.setattr(module, "ant_solution_ACS", lambda *args: next(solutions))

    path, cost, _, epochs = module.ACS(
        GRAPH,
        START_NODE,
        END_NODE,
        1,
        EVAPORATION_RATE,
        LOCAL_EVAPORATION_RATE,
        TRANSITION_PROBABILITY,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        10,
        stagnation_epochs=3,
    )

    assert (path, cost, epochs) == ([0, 1, 3], 2.0, 4)


def test_acs_non_positive_stagnation_limit_runs_until_max_epochs(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_colony_system.ant_colony_system"
    )
    monkeypatch.setattr(
        module, "ant_solution_ACS", lambda *args: ([0, 1, 3], 2.0)
    )

    *_, epochs = module.ACS(
        GRAPH,
        START_NODE,
        END_NODE,
        1,
        EVAPORATION_RATE,
        LOCAL_EVAPORATION_RATE,
        TRANSITION_PROBABILITY,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        3,
        stagnation_epochs=0,
    )

    assert epochs == 3


def test_bwas_uses_finite_worst_and_keeps_global_best(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_best_worst.ant_colony_best_worst"
    )
    captured = {}

    solutions = iter(
        [([0, 1, 3], 2.0), ([0, 2, 3], 5.0), ([0, np.inf], np.inf)]
    )
    monkeypatch.setattr(
        module, "generate_pheromone_map", _capture_generated_pheromones(captured)
    )
    monkeypatch.setattr(
        module, "ant_solution_best_worst", lambda *args: next(solutions)
    )

    path, cost, _, _ = module.ABW(
        GRAPH,
        START_NODE,
        END_NODE,
        ANTS_NUMBER,
        EVAPORATION_RATE,
        1,
        1.0,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        worst_penalty_rate=0.2,
        mutation_probability=0,
    )

    assert (path, cost) == ([0, 1, 3], 2.0)
    assert captured[0][0] == pytest.approx(1.4)
    assert captured[0][1] == pytest.approx(0.72)


def test_bwas_mutates_and_restarts_without_forgetting_best(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_best_worst.ant_colony_best_worst"
    )
    generated_maps = []
    original_generator = module.generate_pheromone_map

    def capture_generator(graph, level):
        pheromones = original_generator(graph, level)
        generated_maps.append(pheromones)
        return pheromones

    solutions = iter(
        [
            ([0, 1, 3], 2.0),
            ([0, 2, 3], 5.0),
            ([0, 2, 3], 5.0),
            ([0, 1, 3], 4.0),
        ]
    )
    monkeypatch.setattr(module, "generate_pheromone_map", capture_generator)
    monkeypatch.setattr(
        module, "ant_solution_best_worst", lambda *args: next(solutions)
    )
    monkeypatch.setattr(module.np.random, "random", lambda: 0.25)
    monkeypatch.setattr(module.np.random, "randint", lambda _: 1)

    path, cost, _, epochs = module.ABW(
        GRAPH,
        START_NODE,
        END_NODE,
        2,
        EVAPORATION_RATE,
        2,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        mutation_probability=1,
        mutation_scale=1,
        restart_stagnation=1,
    )

    assert (path, cost, epochs) == ([0, 1, 3], 2.0, 2)
    assert len(generated_maps) == 2
    assert generated_maps[0][0][1] > 0.082
    assert all(
        np.allclose(values, 0.125) for values in generated_maps[-1].values()
    )


def test_bwas_stopping_patience_does_not_reset_on_restart(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_best_worst.ant_colony_best_worst"
    )
    generated_maps = []
    original_generator = module.generate_pheromone_map

    def capture_generator(graph, level):
        pheromones = original_generator(graph, level)
        generated_maps.append(pheromones)
        return pheromones

    solutions = iter(
        [([0, 1, 3], 2.0)] + [([0, 2, 3], 5.0)] * 3
    )
    monkeypatch.setattr(module, "generate_pheromone_map", capture_generator)
    monkeypatch.setattr(
        module, "ant_solution_best_worst", lambda *args: next(solutions)
    )

    path, cost, _, epochs = module.ABW(
        GRAPH,
        START_NODE,
        END_NODE,
        1,
        EVAPORATION_RATE,
        10,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        mutation_probability=0,
        restart_stagnation=2,
        stagnation_epochs=3,
    )

    assert (path, cost, epochs) == ([0, 1, 3], 2.0, 4)
    assert len(generated_maps) == 2


def test_bwas_non_positive_stagnation_limit_runs_until_max_epochs(monkeypatch):
    module = importlib.import_module(
        "src.scripts.ant_best_worst.ant_colony_best_worst"
    )
    monkeypatch.setattr(
        module, "ant_solution_best_worst", lambda *args: ([0, 1, 3], 2.0)
    )

    *_, epochs = module.ABW(
        GRAPH,
        START_NODE,
        END_NODE,
        1,
        EVAPORATION_RATE,
        3,
        INITIAL_PHEROMONE_LVL,
        HEURISTIC_WEIGHT,
        PHEROMONE_WEIGHT,
        mutation_probability=0,
        restart_stagnation=0,
        stagnation_epochs=0,
    )

    assert epochs == 3


def test_bwas_mutation_uses_search_scaled_direction_and_preserves_f_min(
    monkeypatch,
):
    module = importlib.import_module(
        "src.scripts.ant_best_worst.ant_colony_best_worst"
    )
    pheromones = {0: np.array([1.0, 2.0]), 1: np.array([0.2])}
    directions = iter([1, 0])
    monkeypatch.setattr(module.np.random, "random", lambda: 0.0)
    monkeypatch.setattr(module.np.random, "randint", lambda _: next(directions))

    module._mutate_pheromone_rows(
        pheromones,
        mutation_probability=1.0,
        mutation_scale=0.5,
        current_epoch=5,
        last_restart_epoch=1,
        max_epochs=10,
        global_best_mean=2.0,
        min_pheromone_lvl=0.1,
    )

    # Mutation amount = 0.5 * ((5 - 1) / 10) * 2.0 = 0.4.
    assert pheromones[0] == pytest.approx([1.4, 2.4])
    assert pheromones[1] == pytest.approx([0.1])


@pytest.mark.parametrize(
    ("module_name", "colony_name", "ant_name", "arguments"),
    [
        (
            "src.scripts.ant_colony_simple_ACO.ant_colony_optimization",
            "ACO",
            "ant_solution_ACO",
            (
                START_NODE,
                END_NODE,
                4,
                EVAPORATION_RATE,
                INITIAL_PHEROMONE_LVL,
                HEURISTIC_WEIGHT,
                PHEROMONE_WEIGHT,
                5,
            ),
        ),
        (
            "src.scripts.ant_colony_system.ant_colony_system",
            "ACS",
            "ant_solution_ACS",
            (
                START_NODE,
                END_NODE,
                4,
                EVAPORATION_RATE,
                LOCAL_EVAPORATION_RATE,
                TRANSITION_PROBABILITY,
                INITIAL_PHEROMONE_LVL,
                HEURISTIC_WEIGHT,
                PHEROMONE_WEIGHT,
                5,
            ),
        ),
        (
            "src.scripts.ant_best_worst.ant_colony_best_worst",
            "ABW",
            "ant_solution_best_worst",
            (
                START_NODE,
                END_NODE,
                4,
                EVAPORATION_RATE,
                5,
                INITIAL_PHEROMONE_LVL,
                HEURISTIC_WEIGHT,
                PHEROMONE_WEIGHT,
            ),
        ),
    ],
)
def test_seeded_real_algorithms_return_aligned_best_so_far(
    monkeypatch, module_name, colony_name, ant_name, arguments
):
    np.random.seed(7)
    module = importlib.import_module(module_name)
    real_ant_solution = getattr(module, ant_name)
    observed_solutions = []

    def observe_real_solution(*args):
        solution = real_ant_solution(*args)
        observed_solutions.append(solution)
        return solution

    monkeypatch.setattr(module, ant_name, observe_real_solution)

    path, cost, _, _ = getattr(module, colony_name)(GRAPH, *arguments)

    finite_costs = [
        distance for _, distance in observed_solutions if np.isfinite(distance)
    ]
    assert cost == min(finite_costs)
    assert path[0] == 0
    assert path[-1] == 3
    aligned_cost = sum(
        GRAPH["weights"][current][GRAPH["connections"][current].index(next_node)]
        for current, next_node in zip(path, path[1:])
    )
    assert cost == aligned_cost


@pytest.mark.parametrize(
    ("run_algorithm", "module_name"),
    [
        (_run_aco, "src.scripts.ant_colony_simple_ACO.ant_colony_optimization"),
        (_run_acs, "src.scripts.ant_colony_system.ant_colony_system"),
        (_run_bwas, "src.scripts.ant_best_worst.ant_colony_best_worst"),
    ],
)
def test_seeded_orchestrators_run_end_to_end_on_multimodal_square_graph(
    monkeypatch, run_algorithm, module_name
):
    graph = _multimodal_square_graph()
    module = importlib.import_module(module_name)
    generated = {}
    initial_levels = []
    original_generator = module.generate_pheromone_map

    def capture_generator(graph_map, level):
        initial_levels.append(level)
        pheromones = original_generator(graph_map, level)
        generated.clear()
        generated.update(pheromones)
        return pheromones

    monkeypatch.setattr(module, "generate_pheromone_map", capture_generator)
    np.random.seed(7)

    path, cost, _, epochs = run_algorithm(graph, 0, 2)

    assert path == [
        0,
        "bus:E2E:outbound:0",
        "bus:E2E:outbound:1",
        2,
    ]
    aligned_cost = sum(
        graph["weights"][current][graph["connections"][current].index(next_node)]
        for current, next_node in zip(path, path[1:])
    )
    assert cost == pytest.approx(aligned_cost)
    assert epochs > 0
    assert set(generated) == graph["node_index"]
    assert all(
        len(generated[node]) == len(graph["connections"][node])
        for node in graph["node_index"]
    )
    route_pheromones = [
        generated[current][graph["connections"][current].index(next_node)]
        for current, next_node in zip(path, path[1:])
    ]
    assert all(np.isfinite(route_pheromones))
    assert all(
        pheromone != pytest.approx(initial_levels[0])
        for pheromone in route_pheromones
    )


@pytest.mark.parametrize("run_algorithm", ALGORITHM_RUNNERS)
def test_orchestrators_reject_merged_graph_without_edge_types(run_algorithm):
    graph = _multimodal_square_graph()
    graph.pop("edge_types")

    with pytest.raises(ValueError, match="edge_types"):
        run_algorithm(graph, 0, 2)


@pytest.mark.parametrize("run_algorithm", ALGORITHM_RUNNERS)
def test_trivial_routes_reject_tagged_bus_graph_without_edge_types(run_algorithm):
    bus_node = "bus:line:outbound:0"
    graph = {
        "node_index": {bus_node},
        "connections": {bus_node: []},
        "weights": {bus_node: []},
        "buses": [],
    }

    with pytest.raises(ValueError, match="edge_types"):
        run_algorithm(graph, bus_node, bus_node)


@pytest.mark.parametrize("run_algorithm", ALGORITHM_RUNNERS)
def test_orchestrators_keep_plain_walking_graph_compatibility(run_algorithm):
    graph = {
        "node_index": {0, 1},
        "connections": {0: [1], 1: []},
        "weights": {0: [1.0], 1: []},
    }
    np.random.seed(7)

    path, cost, _, _ = run_algorithm(graph, 0, 1)

    assert path == [0, 1]
    assert cost == pytest.approx(1.0)
