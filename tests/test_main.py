import builtins
import importlib
import runpy
import sys

import pytest

from src.main import _compute_route_cost, _prompt_node, _route_recommendation, prepare_routing_problem


def test_preflight_infers_aligned_walking_types_without_mutating_input():
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [2.0], 1: []}}

  problem = prepare_routing_problem(graph, 0, 1)

  assert problem.graph['edge_types'] == {0: ['walk'], 1: []}
  assert 'edge_types' not in graph


def test_preflight_owns_nested_connections_and_weights():
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [2.0], 1: []}, 'metadata': {'name': 'caller graph'}}

  problem = prepare_routing_problem(graph, 0, 1)
  problem.graph['connections'][0].append(0)
  problem.graph['weights'][0][0] = 99.0

  assert graph['connections'] == {0: [1], 1: []}
  assert graph['weights'] == {0: [2.0], 1: []}
  assert problem.graph['metadata'] == graph['metadata']
  assert problem.graph['connections'] is not graph['connections']
  assert problem.graph['weights'] is not graph['weights']
  assert problem.graph['metadata'] is not graph['metadata']


def test_preflight_validates_endpoints_and_graph_alignment():
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [], 1: []}}

  with pytest.raises(ValueError, match='aligned'):
    prepare_routing_problem(graph, 0, 1)

  valid_graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  with pytest.raises(ValueError, match='must exist'):
    prepare_routing_problem(valid_graph, 0, 99)


def test_routing_problem_owns_trivial_route_or_delegates_without_cost_changes():
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  trivial = prepare_routing_problem(graph, 0, 0)

  assert trivial.run(lambda *_: pytest.fail('algorithm should not run')) == ([0], 0.0, 0.0, 0)

  problem = prepare_routing_problem(graph, 0, 1)
  expected = ([0, 1], 7.25, 0.01, 3)
  assert problem.run(lambda *_args, **_kwargs: expected, ignored=True) == expected


def test_preflight_rejects_multimodal_graph_without_explicit_edge_types():
  bus_node = 'bus:test:outbound:0'
  graph = {'node_index': {0, bus_node}, 'connections': {0: [bus_node], bus_node: []}, 'weights': {0: [1.0], bus_node: []}}

  with pytest.raises(ValueError, match='edge_types'):
    prepare_routing_problem(graph, 0, bus_node)


def test_preflight_preserves_opaque_mixed_ids():
  bus_node = 'vehicle-X'
  graph = {'node_index': {0, bus_node}, 'connections': {0: [bus_node], bus_node: []}, 'weights': {0: [1.0], bus_node: []}, 'edge_types': {0: ['board'], bus_node: []}}

  problem = prepare_routing_problem(graph, 0, bus_node)

  assert problem.graph == graph
  assert problem.graph is not graph
  assert bus_node in problem.graph['node_index']


def test_route_recommendation_compares_numeric_costs_and_handles_missing_routes():
  assert _route_recommendation(3.0, 4.0) == "you'd better take the bus instead of walking."
  assert _route_recommendation(4.0, 4.0) == "you'd better go by foot."
  assert _route_recommendation(float('inf'), 4.0) == "you'd better go by foot."
  assert _route_recommendation(float('inf'), float('inf')) == 'no route could be found.'
  assert _compute_route_cost({}, None) == float('inf')


def test_prompt_node_retries_non_integer_input(monkeypatch, capsys):
  responses = iter(['not-a-number', '4'])
  monkeypatch.setattr(builtins, 'input', lambda _: next(responses))

  assert _prompt_node('Node', 3, 0, 9) == 4
  assert capsys.readouterr().out == 'Please enter a valid integer.\n'


def test_prompt_node_retries_out_of_range_input(monkeypatch, capsys):
  responses = iter(['-1', '10', '5'])
  monkeypatch.setattr(builtins, 'input', lambda _: next(responses))

  assert _prompt_node('Node', 3, 0, 9) == 5
  assert capsys.readouterr().out == 'Please enter a value between 0 and 9.\nPlease enter a value between 0 and 9.\n'


@pytest.mark.parametrize(
  ('module_name', 'module_prefix', 'configuration_module', 'from_src'),
  [('src.main', 'src.scripts', 'src.configuration.algorithm_settings', False), ('main', 'scripts', 'configuration.algorithm_settings', True)],
)
def test_documented_module_forms_run_all_algorithms_without_history_files(monkeypatch, capsys, module_name, module_prefix, configuration_module, from_src):
  if from_src:
    monkeypatch.syspath_prepend('src')

  calls = []
  draw_calls = []

  def algorithm(name):
    def run(graph, start_node, end_node, *args, **kwargs):
      calls.append((name, graph, start_node, end_node, args, kwargs))
      return [start_node, end_node], 1.0, 0.01, 2

    return run

  monkeypatch.setattr(builtins, 'input', lambda _: '')
  monkeypatch.setattr(importlib.import_module(f'{module_prefix}.utils.graph_visualizer'), 'draw_graph', lambda *args, **kwargs: draw_calls.append((args, kwargs)))
  monkeypatch.setattr(importlib.import_module(f'{module_prefix}.ant_colony_simple_ACO.ant_colony_optimization'), 'ACO', algorithm('ACO'))
  monkeypatch.setattr(importlib.import_module(f'{module_prefix}.ant_colony_system.ant_colony_system'), 'ACS', algorithm('ACS'))
  monkeypatch.setattr(importlib.import_module(f'{module_prefix}.ant_best_worst.ant_colony_best_worst'), 'ABW', algorithm('ABW'))
  base_settings = {
    'ants': 17,
    'evaporation_rate': 0.11,
    'local_evaporation_rate': 0.22,
    'transition_probability': 0.33,
    'f_ini': 0.44,
    'alfa': 0.55,
    'beta': 0.66,
    'epomax': 77,
    'global_best_patience': 8,
    'worst_penalty_rate': 0.12,
    'mutation_probability': 0.13,
    'mutation_scale': 1.4,
    'bwas_restart_stagnation': 5,
    'f_min': 0.0002,
  }
  monkeypatch.setattr(importlib.import_module(configuration_module), 'settings', base_settings)
  sys.modules.pop(module_name, None)

  runpy.run_module(module_name, run_name='__main__')

  assert [call[0] for call in calls] == ['ACO', 'ACS', 'ABW']
  assert all((call[2], call[3]) == (3, 69) for call in calls)
  assert calls[0][4:] == (
    (base_settings['ants'], base_settings['evaporation_rate'], base_settings['f_ini'], base_settings['alfa'], base_settings['beta'], base_settings['epomax']),
    {'global_best_patience': base_settings['global_best_patience']},
  )
  assert calls[1][4:] == (
    (
      base_settings['ants'],
      base_settings['evaporation_rate'],
      base_settings['local_evaporation_rate'],
      base_settings['transition_probability'],
      base_settings['f_ini'],
      base_settings['alfa'],
      base_settings['beta'],
      base_settings['epomax'],
    ),
    {'global_best_patience': base_settings['global_best_patience']},
  )
  assert calls[2][4:] == (
    (base_settings['ants'], base_settings['evaporation_rate'], base_settings['epomax'], base_settings['f_ini'], base_settings['alfa'], base_settings['beta']),
    {
      'worst_penalty_rate': base_settings['worst_penalty_rate'],
      'mutation_probability': base_settings['mutation_probability'],
      'mutation_scale': base_settings['mutation_scale'],
      'restart_stagnation': base_settings['bwas_restart_stagnation'],
      'min_pheromone_lvl': base_settings['f_min'],
      'global_best_patience': base_settings['global_best_patience'],
    },
  )
  assert [call[1]['save_path'] for call in draw_calls] == ['toy_city_graph_solution.html', 'toy_city_graph_solution_walking.html']
  output = capsys.readouterr().out
  assert 'Dijkstra walking cost:' in output
  assert all(f'{name} epochs: 2' in output for name in ('ACO', 'ACS', 'ABW'))
