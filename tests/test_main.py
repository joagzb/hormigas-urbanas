import builtins
import importlib
import runpy
import sys
import warnings
from pathlib import Path

import pytest

from src.main import _compute_route_cost, _parse_args, _prompt_node, _route_recommendation, prepare_routing_problem


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


def test_cli_rejects_unknown_preset_with_available_choices(capsys):
  with pytest.raises(SystemExit) as error:
    _parse_args(['--preset', 'not-a-preset'])

  assert error.value.code == 2
  assert 'invalid choice' in capsys.readouterr().err


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
  ('module_name', 'module_prefix', 'configuration_module', 'from_src', 'preset_name'),
  [
    ('src.main', 'src.scripts', 'src.configuration.algorithm_settings', False, None),
    ('src.main', 'src.scripts', 'src.configuration.algorithm_settings', False, 'bus_friendly'),
    ('main', 'scripts', 'configuration.algorithm_settings', True, None),
    ('main', 'scripts', 'configuration.algorithm_settings', True, 'bus_friendly'),
  ],
)
def test_documented_module_forms_forward_settings_filter_output_and_write_route_html(monkeypatch, capsys, module_name, module_prefix, configuration_module, from_src, preset_name):
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
  selected_settings = {**base_settings, 'ants': 29, 'beta': 4.2}
  configuration = importlib.import_module(configuration_module)
  monkeypatch.setattr(configuration, 'settings', base_settings)
  monkeypatch.setattr(configuration, 'load_profile', lambda name: dict(selected_settings))
  monkeypatch.setattr(sys, 'argv', [module_name, *([] if preset_name is None else ['--preset', preset_name])])
  sys.modules.pop(module_name, None)

  runpy.run_module(module_name, run_name='__main__')

  expected_settings = base_settings if preset_name is None else selected_settings
  assert [call[0] for call in calls] == ['ACO', 'ACS', 'ABW']
  assert all((call[2], call[3]) == (3, 69) for call in calls)
  assert calls[0][4:] == (
    (
      expected_settings['ants'],
      expected_settings['evaporation_rate'],
      expected_settings['f_ini'],
      expected_settings['alfa'],
      expected_settings['beta'],
      expected_settings['epomax'],
    ),
    {'global_best_patience': expected_settings['global_best_patience']},
  )
  assert calls[1][4:] == (
    (
      expected_settings['ants'],
      expected_settings['evaporation_rate'],
      expected_settings['local_evaporation_rate'],
      expected_settings['transition_probability'],
      expected_settings['f_ini'],
      expected_settings['alfa'],
      expected_settings['beta'],
      expected_settings['epomax'],
    ),
    {'global_best_patience': expected_settings['global_best_patience']},
  )
  assert calls[2][4:] == (
    (
      expected_settings['ants'],
      expected_settings['evaporation_rate'],
      expected_settings['epomax'],
      expected_settings['f_ini'],
      expected_settings['alfa'],
      expected_settings['beta'],
    ),
    {
      'worst_penalty_rate': expected_settings['worst_penalty_rate'],
      'mutation_probability': expected_settings['mutation_probability'],
      'mutation_scale': expected_settings['mutation_scale'],
      'restart_stagnation': expected_settings['bwas_restart_stagnation'],
      'min_pheromone_lvl': expected_settings['f_min'],
      'global_best_patience': expected_settings['global_best_patience'],
    },
  )
  repository_tmp = Path(__file__).resolve().parents[1] / 'tmp'
  assert [call[1]['save_path'] for call in draw_calls] == [repository_tmp / 'aco_route.html', repository_tmp / 'acs_route.html', repository_tmp / 'bwas_route.html']
  reference_paths = [call[1]['reference_path'] for call in draw_calls]
  assert reference_paths[0][0] == 3 and reference_paths[0][-1] == 69
  assert reference_paths == [reference_paths[0]] * 3
  assert [call[1]['title'] for call in draw_calls] == [
    f'ACO route ({expected_settings["ants"]} ants)',
    f'ACS route ({expected_settings["ants"]} ants)',
    f'BWAS route ({expected_settings["ants"]} ants)',
  ]
  assert [call[1]['route_label'] for call in draw_calls] == [call[1]['title'] for call in draw_calls]
  output = capsys.readouterr().out
  assert all(f'{name} route solution: [3, 69]' in output for name in ('ACO', 'ACS', 'ABW'))
  assert all(f'{name} epochs: 2' in output for name in ('ACO', 'ACS', 'ABW'))
  assert not any(forbidden in output for forbidden in (' time:', 'Dijkstra', 'walking', 'recommendation', 'better'))
  if preset_name is None:
    assert 'Preset:' not in output
  else:
    assert output.index('  ants: 29') < output.index('  beta: 4.2')
    assert output.startswith('Preset: bus_friendly\n')


def test_cli_identical_endpoints_skip_algorithms_and_write_metadata_html(monkeypatch, tmp_path, capsys):
  import src.main as main_module

  responses = iter(['3', '3'])
  monkeypatch.setattr(builtins, 'input', lambda _: next(responses))
  monkeypatch.setattr(main_module, '__file__', str(tmp_path / 'repository' / 'src' / 'main.py'))
  algorithm_code = {main_module.ACO.__code__, main_module.ACS.__code__, main_module.ABW.__code__}
  algorithm_calls = []

  def record_algorithm_calls(frame, event, _arg):
    if event == 'call' and frame.f_code in algorithm_code:
      algorithm_calls.append(frame.f_code.co_name)

  previous_profile = sys.getprofile()
  sys.setprofile(record_algorithm_calls)
  try:
    with warnings.catch_warnings():
      warnings.simplefilter('error', RuntimeWarning)
      main_module.main([])
  finally:
    sys.setprofile(previous_profile)

  assert algorithm_calls == []
  assert capsys.readouterr().out.splitlines() == [
    'ACO route solution: [3]',
    'ACO cost: 0.0',
    'ACO epochs: 0',
    'ACS route solution: [3]',
    'ACS cost: 0.0',
    'ACS epochs: 0',
    'ABW route solution: [3]',
    'ABW cost: 0.0',
    'ABW epochs: 0',
  ]
  output_directory = tmp_path / 'repository' / 'tmp'
  expected_metadata = {
    'aco_route.html': f'ACO route ({main_module.settings["ants"]} ants)',
    'acs_route.html': f'ACS route ({main_module.settings["ants"]} ants)',
    'bwas_route.html': f'BWAS route ({main_module.settings["ants"]} ants)',
  }
  assert {path.name for path in output_directory.iterdir()} == set(expected_metadata)
  for filename, metadata in expected_metadata.items():
    assert metadata in (output_directory / filename).read_text(encoding='utf-8')


def test_bwas_uses_safe_defaults_when_optional_settings_are_absent(monkeypatch):
  import src.main as main_module

  captured = {}
  minimal_settings = {
    'ants': 2,
    'evaporation_rate': 0.1,
    'local_evaporation_rate': 0.1,
    'transition_probability': 0.9,
    'f_ini': None,
    'alfa': 1.0,
    'beta': 2.0,
    'epomax': 3,
    'global_best_patience': 2,
  }

  monkeypatch.setattr(main_module, '_parse_args', lambda argv: type('Args', (), {'preset': None})())
  monkeypatch.setattr(main_module, '_resolved_settings', lambda _: minimal_settings)
  monkeypatch.setattr(main_module, 'generate_square_city_graph', lambda *_: {'map': True})
  monkeypatch.setattr(main_module, 'generate_bus_line_square_city', lambda *_: {'bus': True})
  graph = {'node_index': {3, 69}, 'connections': {3: [69], 69: []}, 'weights': {3: [1.0], 69: []}, 'edge_types': {3: ['walk'], 69: []}}
  monkeypatch.setattr(main_module, 'merge_bus_and_map_graph', lambda *_: graph)
  monkeypatch.setattr(main_module, '_prompt_node', lambda _prompt, default, *_: default)
  monkeypatch.setattr(main_module, 'dijkstra', lambda *_: [3, 69])
  monkeypatch.setattr(main_module, 'ACO', lambda *_args, **_kwargs: ([3, 69], 1.0, 0.0, 1))
  monkeypatch.setattr(main_module, 'ACS', lambda *_args, **_kwargs: ([3, 69], 1.0, 0.0, 1))

  def fake_bwas(*_args, **kwargs):
    captured.update(kwargs)
    return [3, 69], 1.0, 0.0, 1

  monkeypatch.setattr(main_module, 'ABW', fake_bwas)
  monkeypatch.setattr(main_module, '_write_route_html', lambda *_: None)

  main_module.main([])

  assert captured == {
    'worst_penalty_rate': None,
    'mutation_probability': 0.05,
    'mutation_scale': 2.0,
    'restart_stagnation': None,
    'min_pheromone_lvl': None,
    'global_best_patience': 2,
  }
