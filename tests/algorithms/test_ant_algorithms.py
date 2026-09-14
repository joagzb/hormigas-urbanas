import importlib
import inspect
import warnings

import numpy as np
import pytest

from src.scripts.ant_best_worst.ant_solution_ABW import ant_solution_best_worst
from src.scripts.ant_colony_simple_ACO.ant_solution_ACO import ant_solution_ACO
from src.scripts.ant_colony_system.ant_solution_ACS import ant_solution_ACS
from src.scripts.main import prepare_routing_problem
from src.scripts.utils.generators import generate_bus_line_square_city, generate_square_city_graph, merge_bus_and_map_graph
from src.scripts.utils.graph_visualizer import PheromoneHistoryWriter, draw_pheromone_history, load_pheromone_history, stable_edge_order

GRAPH = {
  'node_index': {0, 1, 2, 3},
  'connections': {0: [1, 2], 1: [3], 2: [3], 3: []},
  'weights': {0: [1.0, 4.0], 1: [1.0], 2: [1.0], 3: []},
  'edge_types': {0: ['walk', 'walk'], 1: ['walk'], 2: ['walk'], 3: []},
}

UNREACHABLE_GRAPH = {'node_index': {0, 1, 2}, 'connections': {0: [1], 1: [], 2: []}, 'weights': {0: [1.0], 1: [], 2: []}}

EMPTY_ADJACENCY_GRAPH = {'node_index': {0, 1}, 'connections': {0: [], 1: []}, 'weights': {0: [], 1: []}}

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
    captured.update({node: np.full(len(edges), level) for node, edges in graph['connections'].items()})
    return captured

  return capture_generator


def _run_aco(graph, start_node, end_node):
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')
  problem = prepare_routing_problem(graph, start_node, end_node)
  return problem.run(module.ACO, ANTS_NUMBER, EVAPORATION_RATE, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, MAX_EPOCHS)


def _run_acs(graph, start_node, end_node):
  module = importlib.import_module('src.scripts.ant_colony_system.ant_colony_system')
  problem = prepare_routing_problem(graph, start_node, end_node)
  return problem.run(
    module.ACS, ANTS_NUMBER, EVAPORATION_RATE, LOCAL_EVAPORATION_RATE, TRANSITION_PROBABILITY, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, MAX_EPOCHS
  )


def _run_bwas(graph, start_node, end_node):
  module = importlib.import_module('src.scripts.ant_best_worst.ant_colony_best_worst')
  problem = prepare_routing_problem(graph, start_node, end_node)
  return problem.run(module.ABW, ANTS_NUMBER, EVAPORATION_RATE, MAX_EPOCHS, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT)


ALGORITHM_RUNNERS = [_run_aco, _run_acs, _run_bwas]

PATIENCE_ALGORITHMS = [
  (
    'src.scripts.ant_colony_system.ant_colony_system',
    'ACS',
    'ant_solution_ACS',
    (START_NODE, END_NODE, 1, EVAPORATION_RATE, LOCAL_EVAPORATION_RATE, TRANSITION_PROBABILITY, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT),
  ),
  ('src.scripts.ant_best_worst.ant_colony_best_worst', 'ABW', 'ant_solution_best_worst', (START_NODE, END_NODE, 1, EVAPORATION_RATE)),
]

MAX_EPOCH_ALGORITHMS = [
  (
    'src.scripts.ant_colony_simple_ACO.ant_colony_optimization',
    'ACO',
    'ant_solution_ACO',
    (START_NODE, END_NODE, 1, EVAPORATION_RATE, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 3),
    {},
  ),
  (
    'src.scripts.ant_colony_system.ant_colony_system',
    'ACS',
    'ant_solution_ACS',
    (START_NODE, END_NODE, 1, EVAPORATION_RATE, LOCAL_EVAPORATION_RATE, TRANSITION_PROBABILITY, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 3),
    {},
  ),
  (
    'src.scripts.ant_best_worst.ant_colony_best_worst',
    'ABW',
    'ant_solution_best_worst',
    (START_NODE, END_NODE, 1, EVAPORATION_RATE, 3, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT),
    {'mutation_probability': 0, 'restart_stagnation': 0},
  ),
]


def _multimodal_square_graph():
  map_graph = generate_square_city_graph(2, fixed_weight=10.0)
  bus_services = generate_bus_line_square_city(2, fixed_weight=1.0, line_id='E2E', route=[0, 2])
  return merge_bus_and_map_graph(map_graph, bus_services)


@pytest.mark.parametrize('run_algorithm', ALGORITHM_RUNNERS)
def test_derived_pheromone_returns_trivial_route_when_start_equals_end(run_algorithm):
  path, cost, _, epochs = run_algorithm(EMPTY_ADJACENCY_GRAPH, 0, 0)

  assert (path, cost, epochs) == ([0], 0.0, 0)


@pytest.mark.parametrize('run_algorithm', ALGORITHM_RUNNERS)
def test_trivial_route_requires_existing_node(run_algorithm):
  with pytest.raises(ValueError, match='must exist'):
    run_algorithm(EMPTY_ADJACENCY_GRAPH, 99, 99)


@pytest.mark.parametrize('run_algorithm', ALGORITHM_RUNNERS)
@pytest.mark.parametrize(('graph', 'end_node'), [(UNREACHABLE_GRAPH, 2), (EMPTY_ADJACENCY_GRAPH, 1)])
def test_derived_pheromone_returns_no_route_for_unreachable_graph(run_algorithm, graph, end_node):
  path, cost, _, epochs = run_algorithm(graph, 0, end_node)

  assert path is None
  assert np.isinf(cost)
  assert epochs == 0


@pytest.mark.parametrize(
  ('run_algorithm', 'module_name', 'ant_name', 'expected_tau0'),
  [
    (_run_aco, 'src.scripts.ant_colony_simple_ACO.ant_colony_optimization', 'ant_solution_ACO', 2.0),
    (_run_acs, 'src.scripts.ant_colony_system.ant_colony_system', 'ant_solution_ACS', 0.125),
    (_run_bwas, 'src.scripts.ant_best_worst.ant_colony_best_worst', 'ant_solution_best_worst', 0.125),
  ],
)
def test_derived_pheromone_initialization_does_not_invoke_dijkstra(monkeypatch, run_algorithm, module_name, ant_name, expected_tau0):
  generators = importlib.import_module('src.scripts.utils.generators')
  route_finder = importlib.import_module('src.scripts.utils.dijkstra')
  module = importlib.import_module(module_name)
  generated_levels = []
  original_generator = module.generate_pheromone_map

  def fail_if_called(*args):
    raise AssertionError('Dijkstra must remain an external evaluation oracle')

  def capture_generator(graph, level):
    generated_levels.append(level)
    return original_generator(graph, level)

  monkeypatch.setattr(route_finder, 'dijkstra', fail_if_called)
  monkeypatch.setattr(module, 'generate_pheromone_map', capture_generator)
  monkeypatch.setattr(module, ant_name, lambda *args: ([0, 1, 3], 2.0))

  path, cost, _, _ = run_algorithm(GRAPH, START_NODE, END_NODE)

  assert 'dijkstra' not in inspect.getsource(generators)
  assert 'dijkstra' not in inspect.getsource(module)
  assert generated_levels[0] == pytest.approx(expected_tau0)
  assert (path, cost) == ([0, 1, 3], 2.0)


@pytest.mark.parametrize('ant_solution', [ant_solution_ACO, ant_solution_best_worst])
def test_aco_and_bwas_transitions_use_actual_aligned_edge_cost(monkeypatch, ant_solution):
  observed = {}

  def select_first(probabilities):
    observed.setdefault('probabilities', probabilities)
    return 0

  module = importlib.import_module(ant_solution.__module__)
  monkeypatch.setattr(module, 'roulette_wheel_selection', select_first)
  pheromones = {0: np.array([1.0, 1.0]), 1: np.array([1.0]), 2: np.array([1.0]), 3: np.array([])}

  ant_solution(GRAPH, pheromones, START_NODE, END_NODE, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT)

  assert observed['probabilities'] == pytest.approx([0.8, 0.2])


@pytest.mark.parametrize('ant_solution', [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst])
def test_preflight_graph_supports_direct_ant_construction(ant_solution):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  graph = prepare_routing_problem(graph, 0, 1).graph
  pheromones = {0: np.array([1.0]), 1: np.array([])}
  arguments = [graph, pheromones, 0, 1]
  if ant_solution is ant_solution_ACS:
    arguments.append(TRANSITION_PROBABILITY)
  arguments.extend([HEURISTIC_WEIGHT, PHEROMONE_WEIGHT])

  path, cost = ant_solution(*arguments)

  assert path == [0, 1]
  assert cost == pytest.approx(1.0)


@pytest.mark.parametrize('ant_solution', [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst])
def test_transitions_use_normalized_weights_but_return_original_cost(monkeypatch, ant_solution):
  graph = {'node_index': {0, 1, 2}, 'connections': {0: [1, 2], 1: [], 2: []}, 'weights': {0: [0.3, 0.01], 1: [], 2: []}, 'edge_types': {0: ['ride', 'alight'], 1: [], 2: []}}
  pheromones = {0: np.array([1.0, 1.0]), 1: np.array([]), 2: np.array([])}
  module = importlib.import_module(ant_solution.__module__)
  monkeypatch.setattr(module, 'roulette_wheel_selection', lambda probabilities: int(np.argmax(probabilities)))

  arguments = [graph, pheromones, 0, 1]
  if ant_solution is ant_solution_ACS:
    arguments.append(1.0)
  arguments.extend([1.0, 2.0])
  path, cost = ant_solution(*arguments)

  assert path == [0, 1]
  assert cost == pytest.approx(0.3)


@pytest.mark.parametrize('ant_solution', [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst])
def test_transitions_preserve_opaque_bus_node_ids(monkeypatch, ant_solution):
  bus_node = 'bus:line:outbound:0'
  graph = {
    'node_index': {0, bus_node, 1},
    'connections': {0: [bus_node], bus_node: [1], 1: []},
    'weights': {0: [1.4], bus_node: [0.3], 1: []},
    'edge_types': {0: ['board'], bus_node: ['ride'], 1: []},
  }
  pheromones = {0: np.array([1.0]), bus_node: np.array([1.0]), 1: np.array([])}
  module = importlib.import_module(ant_solution.__module__)
  monkeypatch.setattr(module, 'roulette_wheel_selection', lambda probabilities: 0)

  arguments = [graph, pheromones, 0, 1]
  if ant_solution is ant_solution_ACS:
    arguments.append(0.0)
  arguments.extend([1.0, 1.0])
  path, cost = ant_solution(*arguments)

  assert path == [0, bus_node, 1]
  assert cost == pytest.approx(1.7)


@pytest.mark.parametrize('ant_solution', [ant_solution_ACO, ant_solution_ACS, ant_solution_best_worst])
def test_mixed_node_ids_do_not_allow_route_revisits(monkeypatch, ant_solution):
  bus_node = 'bus:cycle:outbound:0'
  graph = {
    'node_index': {0, bus_node, 1, 2},
    'connections': {0: [bus_node], bus_node: [0, 1], 1: [bus_node, 2], 2: []},
    'weights': {0: [1.0], bus_node: [0.1, 1.0], 1: [0.1, 1.0], 2: []},
    'edge_types': {0: ['board'], bus_node: ['alight', 'ride'], 1: ['board', 'walk'], 2: []},
  }
  pheromones = {node: np.ones(len(connections)) for node, connections in graph['connections'].items()}
  selection_count = 0

  def select_first(probabilities):
    nonlocal selection_count
    selection_count += 1
    if selection_count > len(graph['node_index']):
      raise RuntimeError('route construction exceeded graph size')
    return 0

  module = importlib.import_module(ant_solution.__module__)
  monkeypatch.setattr(module, 'roulette_wheel_selection', select_first)
  arguments = [graph, pheromones, 0, 2]
  if ant_solution is ant_solution_ACS:
    monkeypatch.setattr(module.np.random, 'rand', lambda: 1.0)
    arguments.append(0.0)
  arguments.extend([1.0, 1.0])

  path, cost = ant_solution(*arguments)

  assert path == [0, bus_node, 1, 2]
  assert len(path) <= len(graph['node_index'])
  assert len(path) == len(set(path))
  aligned_cost = sum(graph['weights'][current][graph['connections'][current].index(next_node)] for current, next_node in zip(path, path[1:]))
  assert cost == pytest.approx(aligned_cost)


def test_aco_derives_tau0_and_retains_best_so_far(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')
  generated_levels = []
  original_generator = module.generate_pheromone_map

  def capture_generator(graph, level):
    generated_levels.append(level)
    return original_generator(graph, level)

  solutions = iter([([0, 1, 3], 2.0), ([0, 2, 3], 5.0), ([0, 2, 3], 5.0), ([0, 1, 3], 4.0)])
  monkeypatch.setattr(module, 'generate_pheromone_map', capture_generator)
  monkeypatch.setattr(module, 'ant_solution_ACO', lambda *args: next(solutions))

  path, cost, _, epochs = module.ACO(GRAPH, START_NODE, END_NODE, 2, EVAPORATION_RATE, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 2)

  assert generated_levels == [2.0]
  assert (path, cost, epochs) == ([0, 1, 3], 2.0, 2)


def test_aco_strict_global_best_improvement_resets_patience(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')
  solutions = iter([([0, 2, 3], 5.0), ([0, 2, 3], 5.0), ([0, 1, 3], 2.0), ([0, 2, 3], 5.0), ([0, 2, 3], 5.0)])
  monkeypatch.setattr(module, 'ant_solution_ACO', lambda *args: next(solutions))

  path, cost, _, epochs = module.ACO(GRAPH, START_NODE, END_NODE, 1, EVAPORATION_RATE, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 8, global_best_patience=2)

  assert (path, cost, epochs) == ([0, 1, 3], 2.0, 5)


def test_aco_no_improvement_stops_after_configured_patience(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')
  monkeypatch.setattr(module, 'ant_solution_ACO', lambda *args: ([0, 1, 3], 2.0))

  *_, epochs = module.ACO(GRAPH, START_NODE, END_NODE, 1, EVAPORATION_RATE, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 10, global_best_patience=3)

  assert epochs == 4


def test_aco_patience_cannot_stop_before_epoch_two(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')
  monkeypatch.setattr(module, 'ant_solution_ACO', lambda *args: ([0, 1, 3], 2.0))

  *_, epochs = module.ACO(GRAPH, START_NODE, END_NODE, 1, EVAPORATION_RATE, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 5, global_best_patience=1)

  assert epochs == 2


def test_aco_no_finite_route_stops_when_patience_is_exhausted(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')
  monkeypatch.setattr(module, 'ant_solution_ACO', lambda *args: (None, np.inf))

  path, cost, _, epochs = module.ACO(GRAPH, START_NODE, END_NODE, 1, EVAPORATION_RATE, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 4, global_best_patience=1)

  assert path is None
  assert np.isinf(cost)
  assert epochs == 1


def test_aco_callback_observes_stopping_epoch_before_break(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')
  monkeypatch.setattr(module, 'ant_solution_ACO', lambda *args: ([0, 1, 3], 2.0))
  observations = []

  *_, epochs = module.ACO(GRAPH, START_NODE, END_NODE, 1, EVAPORATION_RATE, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 5, global_best_patience=1, epoch_callback=observations.append)

  assert epochs == 2
  assert [observation['epoch'] for observation in observations] == [1, 2]
  assert observations[-1]['stage'] == 'pheromone_update'


@pytest.mark.parametrize('patience', [0, -1, 1.5, True])
def test_aco_rejects_invalid_global_best_patience(patience):
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')

  with pytest.raises(ValueError, match='positive integer'):
    module.ACO(GRAPH, START_NODE, END_NODE, 1, EVAPORATION_RATE, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 2, global_best_patience=patience)


def test_acs_updates_selected_edge_immediately_toward_tau0(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_system.ant_solution_ACS')
  monkeypatch.setattr(module.np.random, 'rand', lambda: 0.0)
  pheromones = {0: np.array([1.0, 1.0]), 1: np.array([1.0]), 2: np.array([1.0]), 3: np.array([])}

  path, _ = ant_solution_ACS(GRAPH, pheromones, START_NODE, END_NODE, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 0.5, 0.2)

  assert path == [0, 1, 3]
  assert pheromones[0][0] == pytest.approx(0.6)
  assert pheromones[1][0] == pytest.approx(0.6)
  assert pheromones[0][1] == pytest.approx(1.0)


def test_acs_roulette_uses_zero_based_second_neighbor(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_system.ant_solution_ACS')
  graph = {'node_index': {0, 1, 2}, 'connections': {0: [1, 2], 1: [], 2: []}, 'weights': {0: [1.0, 2.0], 1: [], 2: []}, 'edge_types': {0: ['walk', 'walk'], 1: [], 2: []}}
  pheromones = {0: np.array([1.0, 1.0]), 1: np.array([]), 2: np.array([])}
  monkeypatch.setattr(module.np.random, 'rand', lambda: 1.0)
  monkeypatch.setattr(module.np.random, 'choice', lambda class_count, p: 1)

  path, cost = ant_solution_ACS(graph, pheromones, 0, 2, 0.0, 1.0, 1.0)

  assert path == [0, 2]
  assert cost == pytest.approx(2.0)


def test_acs_global_update_only_touches_retained_global_best(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_system.ant_colony_system')
  captured = {}

  solutions = iter([([0, 1, 3], 2.0), ([0, 2, 3], 5.0)])
  monkeypatch.setattr(module, 'generate_pheromone_map', _capture_generated_pheromones(captured))
  monkeypatch.setattr(module, 'ant_solution_ACS', lambda *args: next(solutions))

  path, cost, _, _ = module.ACS(GRAPH, START_NODE, END_NODE, 2, 0.5, LOCAL_EVAPORATION_RATE, 1, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 1)

  assert (path, cost) == ([0, 1, 3], 2.0)
  assert captured[0][0] == pytest.approx(0.3125)
  assert captured[1][0] == pytest.approx(0.3125)
  assert captured[0][1] == pytest.approx(0.125)
  assert captured[2][0] == pytest.approx(0.125)


@pytest.mark.parametrize(
  ('module_name', 'colony_name', 'ant_name', 'arguments', 'kwargs'),
  [
    ('src.scripts.ant_colony_simple_ACO.ant_colony_optimization', 'ACO', 'ant_solution_ACO', (0, 1, 1, 0.1, 1.0, 1, 1, 1), {}),
    ('src.scripts.ant_colony_system.ant_colony_system', 'ACS', 'ant_solution_ACS', (0, 1, 1, 0.1, 0.1, 1.0, 1.0, 1, 1, 1), {}),
    ('src.scripts.ant_best_worst.ant_colony_best_worst', 'ABW', 'ant_solution_best_worst', (0, 1, 1, 0.1, 1, 1.0, 1, 1), {'mutation_probability': 0}),
  ],
)
def test_zero_cost_baseline_uses_finite_tau0_and_zero_deposit(monkeypatch, module_name, colony_name, ant_name, arguments, kwargs):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [0.0], 1: []}}
  module = importlib.import_module(module_name)
  captured = {}
  generated_levels = []

  def capture_generator(graph_map, level):
    generated_levels.append(level)
    return _capture_generated_pheromones(captured)(graph_map, level)

  monkeypatch.setattr(module, 'generate_pheromone_map', capture_generator)
  monkeypatch.setattr(module, ant_name, lambda *args: ([0, 1], 0.0))

  arguments = list(arguments)
  initial_pheromone_index = {'ACO': 4, 'ACS': 6, 'ABW': 5}[colony_name]
  arguments[initial_pheromone_index] = None

  with warnings.catch_warnings():
    warnings.simplefilter('error', RuntimeWarning)
    path, cost, _, _ = getattr(module, colony_name)(graph, *arguments, **kwargs)

  assert (path, cost) == ([0, 1], 0.0)
  assert generated_levels == [1.0]
  assert captured[0] == pytest.approx([0.9])
  assert np.isfinite(captured[0]).all()


def test_acs_returns_best_route_seen_across_epochs(monkeypatch):
  module = importlib.import_module('src.scripts.ant_colony_system.ant_colony_system')
  solutions = iter([([0, 1, 3], 2.0), ([0, 2, 3], 5.0), ([0, 2, 3], 5.0), ([0, 1, 3], 4.0)])
  monkeypatch.setattr(module, 'ant_solution_ACS', lambda *args: next(solutions))

  path, cost, _, epochs = module.ACS(GRAPH, START_NODE, END_NODE, 2, EVAPORATION_RATE, LOCAL_EVAPORATION_RATE, 1, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 2)

  assert (path, cost, epochs) == ([0, 1, 3], 2.0, 2)


@pytest.mark.parametrize(('module_name', 'colony_name', 'ant_name', 'arguments'), PATIENCE_ALGORITHMS)
def test_all_variants_stop_after_strict_global_best_patience(monkeypatch, module_name, colony_name, ant_name, arguments):
  module = importlib.import_module(module_name)
  solutions = iter([([0, 1, 3], 3.0), ([0, 1, 3], 2.0), ([0, 1, 3], 2.0), ([0, 1, 3], 2.0)])
  monkeypatch.setattr(module, ant_name, lambda *args: next(solutions))
  if colony_name == 'ABW':
    arguments = (*arguments, 5, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT)
    kwargs = {'mutation_probability': 0, 'restart_stagnation': 0}
  else:
    arguments = (*arguments, 5)
    kwargs = {}

  *_, epochs = getattr(module, colony_name)(GRAPH, *arguments, global_best_patience=2, **kwargs)

  assert epochs == 4


@pytest.mark.parametrize(('module_name', 'colony_name', 'ant_name', 'arguments', 'kwargs'), MAX_EPOCH_ALGORITHMS)
def test_all_variants_stop_at_max_epochs_while_global_best_keeps_improving(monkeypatch, module_name, colony_name, ant_name, arguments, kwargs):
  module = importlib.import_module(module_name)
  observed_costs = []
  improving_costs = iter([3.0, 2.0, 1.0])

  def improving_solution(*args):
    cost = next(improving_costs)
    observed_costs.append(cost)
    return [0, 1, 3], cost

  monkeypatch.setattr(module, ant_name, improving_solution)

  path, cost, _, epochs = getattr(module, colony_name)(GRAPH, *arguments, global_best_patience=1, **kwargs)

  assert (path, cost, epochs) == ([0, 1, 3], 1.0, 3)
  assert observed_costs == [3.0, 2.0, 1.0]


@pytest.mark.parametrize(('module_name', 'colony_name', 'ant_name', 'arguments'), PATIENCE_ALGORITHMS)
def test_all_variants_reject_invalid_global_best_patience(monkeypatch, module_name, colony_name, ant_name, arguments):
  module = importlib.import_module(module_name)
  if colony_name == 'ABW':
    arguments = (*arguments, 1, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT)
  else:
    arguments = (*arguments, 1)

  with pytest.raises(ValueError, match='positive integer'):
    getattr(module, colony_name)(GRAPH, *arguments, global_best_patience=0)


def test_bwas_uses_finite_worst_and_keeps_global_best(monkeypatch):
  module = importlib.import_module('src.scripts.ant_best_worst.ant_colony_best_worst')
  captured = {}

  solutions = iter([([0, 1, 3], 2.0), ([0, 2, 3], 5.0), ([0, np.inf], np.inf)])
  monkeypatch.setattr(module, 'generate_pheromone_map', _capture_generated_pheromones(captured))
  monkeypatch.setattr(module, 'ant_solution_best_worst', lambda *args: next(solutions))

  path, cost, _, _ = module.ABW(
    GRAPH, START_NODE, END_NODE, ANTS_NUMBER, EVAPORATION_RATE, 1, 1.0, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, worst_penalty_rate=0.2, mutation_probability=0
  )

  assert (path, cost) == ([0, 1, 3], 2.0)
  assert captured[0][0] == pytest.approx(1.4)
  assert captured[0][1] == pytest.approx(0.72)


def test_bwas_mutates_and_restarts_without_forgetting_best(monkeypatch):
  module = importlib.import_module('src.scripts.ant_best_worst.ant_colony_best_worst')
  generated_maps = []
  original_generator = module.generate_pheromone_map

  def capture_generator(graph, level):
    pheromones = original_generator(graph, level)
    generated_maps.append(pheromones)
    return pheromones

  solutions = iter([([0, 1, 3], 2.0), ([0, 2, 3], 5.0), ([0, 2, 3], 5.0), ([0, 1, 3], 4.0)])
  monkeypatch.setattr(module, 'generate_pheromone_map', capture_generator)
  monkeypatch.setattr(module, 'ant_solution_best_worst', lambda *args: next(solutions))
  monkeypatch.setattr(module.np.random, 'random', lambda: 0.25)
  monkeypatch.setattr(module.np.random, 'randint', lambda _: 1)

  path, cost, _, epochs = module.ABW(
    GRAPH, START_NODE, END_NODE, 2, EVAPORATION_RATE, 2, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, mutation_probability=1, mutation_scale=1, restart_stagnation=1
  )

  assert (path, cost, epochs) == ([0, 1, 3], 2.0, 2)
  assert len(generated_maps) == 2
  assert generated_maps[0][0][1] > 0.082
  assert all(np.allclose(values, 0.125) for values in generated_maps[-1].values())


def test_bwas_mutation_uses_search_scaled_direction_and_preserves_f_min(monkeypatch):
  module = importlib.import_module('src.scripts.ant_best_worst.ant_colony_best_worst')
  pheromones = {0: np.array([1.0, 2.0]), 1: np.array([0.2])}
  directions = iter([1, 0])
  monkeypatch.setattr(module.np.random, 'random', lambda: 0.0)
  monkeypatch.setattr(module.np.random, 'randint', lambda _: next(directions))

  module._mutate_pheromone_rows(
    pheromones, mutation_probability=1.0, mutation_scale=0.5, current_epoch=5, last_restart_epoch=1, max_epochs=10, global_best_mean=2.0, min_pheromone_lvl=0.1
  )

  # Mutation amount = 0.5 * ((5 - 1) / 10) * 2.0 = 0.4.
  assert pheromones[0] == pytest.approx([1.4, 2.4])
  assert pheromones[1] == pytest.approx([0.1])


@pytest.mark.parametrize(
  ('module_name', 'colony_name', 'ant_name', 'arguments'),
  [
    (
      'src.scripts.ant_colony_simple_ACO.ant_colony_optimization',
      'ACO',
      'ant_solution_ACO',
      (START_NODE, END_NODE, 4, EVAPORATION_RATE, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 5),
    ),
    (
      'src.scripts.ant_colony_system.ant_colony_system',
      'ACS',
      'ant_solution_ACS',
      (START_NODE, END_NODE, 4, EVAPORATION_RATE, LOCAL_EVAPORATION_RATE, TRANSITION_PROBABILITY, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 5),
    ),
    (
      'src.scripts.ant_best_worst.ant_colony_best_worst',
      'ABW',
      'ant_solution_best_worst',
      (START_NODE, END_NODE, 4, EVAPORATION_RATE, 5, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT),
    ),
  ],
)
def test_seeded_real_algorithms_return_aligned_best_so_far(monkeypatch, module_name, colony_name, ant_name, arguments):
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

  finite_costs = [distance for _, distance in observed_solutions if np.isfinite(distance)]
  assert cost == min(finite_costs)
  assert path[0] == 0
  assert path[-1] == 3
  aligned_cost = sum(GRAPH['weights'][current][GRAPH['connections'][current].index(next_node)] for current, next_node in zip(path, path[1:]))
  assert cost == aligned_cost


@pytest.mark.parametrize(
  ('run_algorithm', 'module_name'),
  [
    (_run_aco, 'src.scripts.ant_colony_simple_ACO.ant_colony_optimization'),
    (_run_acs, 'src.scripts.ant_colony_system.ant_colony_system'),
    (_run_bwas, 'src.scripts.ant_best_worst.ant_colony_best_worst'),
  ],
)
def test_seeded_orchestrators_run_end_to_end_on_multimodal_square_graph(monkeypatch, run_algorithm, module_name):
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

  monkeypatch.setattr(module, 'generate_pheromone_map', capture_generator)
  np.random.seed(7)

  path, cost, _, epochs = run_algorithm(graph, 0, 2)

  assert path == [0, 'bus:E2E:outbound:0', 'bus:E2E:outbound:1', 2]
  aligned_cost = sum(graph['weights'][current][graph['connections'][current].index(next_node)] for current, next_node in zip(path, path[1:]))
  assert cost == pytest.approx(aligned_cost)
  assert epochs > 0
  assert set(generated) == graph['node_index']
  assert all(len(generated[node]) == len(graph['connections'][node]) for node in graph['node_index'])
  route_pheromones = [generated[current][graph['connections'][current].index(next_node)] for current, next_node in zip(path, path[1:])]
  assert all(np.isfinite(route_pheromones))
  assert all(pheromone != pytest.approx(initial_levels[0]) for pheromone in route_pheromones)


def test_preflight_rejects_merged_graph_without_edge_types():
  graph = _multimodal_square_graph()
  graph.pop('edge_types')

  with pytest.raises(ValueError, match='edge_types'):
    prepare_routing_problem(graph, 0, 2)


def test_preflight_rejects_tagged_bus_graph_without_edge_types():
  bus_node = 'bus:line:outbound:0'
  graph = {'node_index': {bus_node}, 'connections': {bus_node: []}, 'weights': {bus_node: []}, 'buses': []}

  with pytest.raises(ValueError, match='edge_types'):
    prepare_routing_problem(graph, bus_node, bus_node)


@pytest.mark.parametrize('run_algorithm', ALGORITHM_RUNNERS)
def test_orchestrators_keep_plain_walking_graph_compatibility(run_algorithm):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  np.random.seed(7)

  path, cost, _, _ = run_algorithm(graph, 0, 1)

  assert path == [0, 1]
  assert cost == pytest.approx(1.0)


@pytest.mark.parametrize(
  ('module_name', 'colony_name', 'ant_name', 'arguments', 'kwargs'),
  [
    ('src.scripts.ant_colony_simple_ACO.ant_colony_optimization', 'ACO', 'ant_solution_ACO', (0, 3, 1, 0.1, 1.0, 1, 1, 2), {}),
    ('src.scripts.ant_colony_system.ant_colony_system', 'ACS', 'ant_solution_ACS', (0, 3, 1, 0.1, 0.1, 1.0, 1.0, 1, 1, 2), {}),
    ('src.scripts.ant_best_worst.ant_colony_best_worst', 'ABW', 'ant_solution_best_worst', (0, 3, 1, 0.1, 2, 1.0, 1, 1), {'mutation_probability': 1, 'restart_stagnation': 0}),
  ],
)
def test_orchestrators_emit_live_safe_epoch_callbacks_without_changing_return_tuple(monkeypatch, module_name, colony_name, ant_name, arguments, kwargs):
  module = importlib.import_module(module_name)
  monkeypatch.setattr(module, ant_name, lambda *args: ([0, 1, 3], 2.0))
  final_pheromones = {}
  monkeypatch.setattr(module, 'generate_pheromone_map', _capture_generated_pheromones(final_pheromones))
  observations = []

  result = getattr(module, colony_name)(GRAPH, *arguments, epoch_callback=observations.append, **kwargs)

  assert len(result) == 4
  assert [observation['epoch'] for observation in observations] == list(range(1, result[3] + 1))
  assert all(observation['iteration_best_path'] == [0, 1, 3] for observation in observations)
  assert all(observation['global_best_path'] == [0, 1, 3] for observation in observations)
  assert all(observation['stage'] == 'pheromone_update' for observation in observations)
  for node in GRAPH['node_index']:
    assert np.array_equal(observations[-1]['pheromones'][node], final_pheromones[node])
  if len(observations) > 1:
    observations[0]['pheromones'][0][0] = 999
    assert observations[1]['pheromones'][0][0] != 999


def test_bwas_history_records_final_post_restart_pheromones(monkeypatch, tmp_path):
  module = importlib.import_module('src.scripts.ant_best_worst.ant_colony_best_worst')
  solutions = iter([([0, 1, 3], 2.0), ([0, 2, 3], 5.0)])
  monkeypatch.setattr(module, 'ant_solution_best_worst', lambda *args: next(solutions))
  history_path = tmp_path / 'bwas_history.jsonl'
  writer = PheromoneHistoryWriter(GRAPH, history_path)

  module.ABW(GRAPH, 0, 3, 1, 0.1, 2, 0.5, 1, 1, mutation_probability=0, restart_stagnation=1, epoch_callback=writer)

  restart_observation = load_pheromone_history(GRAPH, history_path)[-1]
  assert restart_observation['stage'] == 'pheromone_update'
  assert restart_observation['restarted'] is True
  assert restart_observation['pheromones'] == pytest.approx([0.5] * 4)


def test_real_aco_file_history_matches_final_state_and_limits_frames(monkeypatch, tmp_path):
  np.random.seed(7)
  module = importlib.import_module('src.scripts.ant_colony_simple_ACO.ant_colony_optimization')
  final_pheromones = {}
  monkeypatch.setattr(module, 'generate_pheromone_map', _capture_generated_pheromones(final_pheromones))
  history_path = tmp_path / 'aco_history.jsonl'
  writer = PheromoneHistoryWriter(GRAPH, history_path)

  path, cost, _, epochs = module.ACO(GRAPH, START_NODE, END_NODE, 3, EVAPORATION_RATE, INITIAL_PHEROMONE_LVL, HEURISTIC_WEIGHT, PHEROMONE_WEIGHT, 5, epoch_callback=writer)
  snapshots = load_pheromone_history(GRAPH, history_path, stride=2, max_frames=4)
  figure = draw_pheromone_history(GRAPH, history_path, reference_path=[0, 1, 3], stride=2, max_frames=4, show=False)
  expected_final = [final_pheromones[edge.source][edge.adjacency_index] for edge in stable_edge_order(GRAPH)]

  assert path is not None
  assert np.isfinite(cost)
  assert snapshots[-1]['epoch'] == epochs
  assert snapshots[-1]['pheromones'] == pytest.approx(expected_final)
  assert len(history_path.read_text(encoding='utf-8').splitlines()) == epochs + 1
  assert len(figure.frames) == len(snapshots) <= 4
  assert figure.layout.sliders[0].steps[-1].label == f'Iteration {epochs}'
