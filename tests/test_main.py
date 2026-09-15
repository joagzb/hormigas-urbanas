import builtins
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.main import _parse_args, _prompt_node, _resolved_settings, prepare_routing_problem


SIMPLE_GRAPH = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}, 'edge_types': {0: ['walk'], 1: []}}


def test_preflight_infers_aligned_walking_types_without_mutating_input():
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [2.0], 1: []}}

  problem = prepare_routing_problem(graph, 0, 1)

  assert problem.graph['edge_types'] == {0: ['walk'], 1: []}
  assert 'edge_types' not in graph


def test_preflight_owns_nested_graph_data():
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [2.0], 1: []}, 'metadata': {'name': 'caller graph'}}

  problem = prepare_routing_problem(graph, 0, 1)
  problem.graph['connections'][0].append(0)
  problem.graph['weights'][0][0] = 99.0
  problem.graph['metadata']['name'] = 'changed'

  assert graph['connections'] == {0: [1], 1: []}
  assert graph['weights'] == {0: [2.0], 1: []}
  assert graph['metadata'] == {'name': 'caller graph'}


def test_preflight_validates_alignment_endpoints_and_multimodal_edge_types():
  misaligned = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [], 1: []}}
  with pytest.raises(ValueError, match='aligned'):
    prepare_routing_problem(misaligned, 0, 1)

  walking = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  with pytest.raises(ValueError, match='must exist'):
    prepare_routing_problem(walking, 0, 99)

  bus_node = 'bus:test:outbound:0'
  multimodal = {'node_index': {0, bus_node}, 'connections': {0: [bus_node], bus_node: []}, 'weights': {0: [1.0], bus_node: []}}
  with pytest.raises(ValueError, match='edge_types'):
    prepare_routing_problem(multimodal, 0, bus_node)


def test_preflight_preserves_opaque_mixed_ids():
  bus_node = 'vehicle-X'
  graph = {'node_index': {0, bus_node}, 'connections': {0: [bus_node], bus_node: []}, 'weights': {0: [1.0], bus_node: []}, 'edge_types': {0: ['board'], bus_node: []}}

  problem = prepare_routing_problem(graph, 0, bus_node)

  assert problem.graph == graph
  assert problem.graph is not graph


def test_default_settings_are_a_fresh_copy():
  first = _resolved_settings(None)
  second = _resolved_settings(None)

  first['ants'] = -1

  assert first is not second
  assert second['ants'] != -1


def test_cli_rejects_unknown_preset_with_available_choices(capsys):
  with pytest.raises(SystemExit) as error:
    _parse_args(['--preset', 'not-a-preset'])

  assert error.value.code == 2
  assert 'invalid choice' in capsys.readouterr().err


def test_prompt_node_retries_invalid_input(monkeypatch, capsys):
  responses = iter(['not-a-number', '-1', '10', '5'])
  monkeypatch.setattr(builtins, 'input', lambda _: next(responses))

  assert _prompt_node('Node', 3, 0, 9) == 5
  assert capsys.readouterr().out.splitlines() == ['Please enter a valid integer.', 'Please enter a value between 0 and 9.', 'Please enter a value between 0 and 9.']


def _import_entrypoint(monkeypatch, module_name):
  if module_name == 'main':
    monkeypatch.syspath_prepend('src')
  sys.modules.pop(module_name, None)
  return importlib.import_module(module_name)


def _configure_main(monkeypatch, module, *, selected=False, writer_class=None):
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
  resolved_settings = {**base_settings, 'ants': 29, 'beta': 4.2} if selected else base_settings
  monkeypatch.setattr(module, 'settings', base_settings)
  monkeypatch.setattr(module, 'load_profile', lambda name: dict(resolved_settings))
  monkeypatch.setattr(module, 'generate_square_city_graph', lambda *_: {'map': True})
  monkeypatch.setattr(module, 'generate_bus_line_square_city', lambda *_: {'bus': True})
  monkeypatch.setattr(module, 'merge_bus_and_map_graph', lambda *_: SIMPLE_GRAPH)
  endpoints = iter([0, 1])
  monkeypatch.setattr(module, '_prompt_node', lambda *_: next(endpoints))
  monkeypatch.setattr(module, 'dijkstra', lambda *_: [0, 1])
  monkeypatch.setattr(module, 'draw_graph', lambda *_args, **_kwargs: None)
  if writer_class is not None:
    monkeypatch.setattr(module, 'ExperimentOutputWriter', writer_class)
  return resolved_settings


@pytest.mark.parametrize('module_name', ['src.main', 'main'])
@pytest.mark.parametrize('preset_name', [None, 'bus_friendly'])
def test_documented_execution_contexts_forward_settings_callbacks_and_animations(monkeypatch, capsys, module_name, preset_name):
  module = _import_entrypoint(monkeypatch, module_name)
  calls = []
  draw_calls = []
  writers = []
  reference_route = [0, 2, 1]

  class FakeWriter:
    def __init__(self, graph, name):
      self.graph = graph
      self.name = name
      self.observations = []
      self.animation_calls = []
      writers.append(self)

    def __call__(self, observation):
      self.observations.append(observation)

    def write_animation(self, **kwargs):
      self.animation_calls.append(kwargs)
      return Path('tmp') / f'{self.name}_pheromone_animation.html'

  def algorithm(name):
    def run(graph, start_node, end_node, *args, **kwargs):
      calls.append((name, graph, start_node, end_node, args, kwargs))
      for epoch in (1, 2):
        kwargs['epoch_callback']({'epoch': epoch})
      return [start_node, end_node], 1.0, 0.01, 2

    return run

  resolved_settings = _configure_main(monkeypatch, module, selected=preset_name is not None, writer_class=FakeWriter)
  monkeypatch.setattr(module, 'dijkstra', lambda *_: reference_route)
  monkeypatch.setattr(module, 'draw_graph', lambda *args, **kwargs: draw_calls.append((args, kwargs)))
  monkeypatch.setattr(module, 'ACO', algorithm('ACO'))
  monkeypatch.setattr(module, 'ACS', algorithm('ACS'))
  monkeypatch.setattr(module, 'ABW', algorithm('ABW'))

  module.main([] if preset_name is None else ['--preset', preset_name])

  assert [call[0] for call in calls] == ['ACO', 'ACS', 'ABW']
  assert all((call[2], call[3]) == (0, 1) for call in calls)
  assert [writer.name for writer in writers] == ['aco', 'acs', 'bwas']
  assert [[item['epoch'] for item in writer.observations] for writer in writers] == [[1, 2], [1, 2], [1, 2]]
  assert [call[5]['epoch_callback'] for call in calls] == writers
  assert calls[0][4] == (
    resolved_settings['ants'],
    resolved_settings['evaporation_rate'],
    resolved_settings['f_ini'],
    resolved_settings['alfa'],
    resolved_settings['beta'],
    resolved_settings['epomax'],
  )
  assert calls[1][4] == (
    resolved_settings['ants'],
    resolved_settings['evaporation_rate'],
    resolved_settings['local_evaporation_rate'],
    resolved_settings['transition_probability'],
    resolved_settings['f_ini'],
    resolved_settings['alfa'],
    resolved_settings['beta'],
    resolved_settings['epomax'],
  )
  assert calls[2][4] == (
    resolved_settings['ants'],
    resolved_settings['evaporation_rate'],
    resolved_settings['epomax'],
    resolved_settings['f_ini'],
    resolved_settings['alfa'],
    resolved_settings['beta'],
  )
  repository_tmp = Path(__file__).resolve().parents[1] / 'tmp'
  assert [call[1]['save_path'] for call in draw_calls] == [repository_tmp / 'aco_route.html', repository_tmp / 'acs_route.html', repository_tmp / 'bwas_route.html']
  assert [call[0][1] for call in draw_calls] == [[0, 1]] * 3
  assert [call[1]['reference_path'] for call in draw_calls] == [reference_route] * 3
  expected_route_metadata = [f'ACO route ({resolved_settings["ants"]} ants)', f'ACS route ({resolved_settings["ants"]} ants)', f'BWAS route ({resolved_settings["ants"]} ants)']
  assert [call[1]['title'] for call in draw_calls] == expected_route_metadata
  assert [call[1]['route_label'] for call in draw_calls] == expected_route_metadata
  assert [writer.animation_calls[0]['reference_path'] for writer in writers] == [reference_route] * 3
  assert [writer.animation_calls[0]['title'] for writer in writers] == ['ACO pheromone evolution', 'ACS pheromone evolution', 'BWAS pheromone evolution']
  assert all(f'{resolved_settings["ants"]} ants' in writer.animation_calls[0]['route_label'] for writer in writers)
  output = capsys.readouterr().out
  assert all(f'{name} route solution: [0, 1]' in output for name in ('ACO', 'ACS', 'ABW'))
  assert all(f'{name} cost: 1.0' in output and f'{name} epochs: 2' in output for name in ('ACO', 'ACS', 'ABW'))
  assert not any(forbidden in output for forbidden in (' time:', 'Dijkstra', 'walking', 'recommendation', 'better'))
  if preset_name is None:
    assert 'Preset:' not in output
    assert '  ants:' not in output
  else:
    assert output.startswith(f'Preset: {preset_name}\n')
    assert output.index('  ants: 29') < output.index('  beta: 4.2')


def test_main_writes_multi_epoch_histories_and_animation_html_under_repository_tmp(monkeypatch, tmp_path):
  import src.main as module
  import src.scripts.utils.graph_visualizer as graph_visualizer

  repository = tmp_path / 'repository'
  monkeypatch.setattr(module, '__file__', str(repository / 'src' / 'main.py'))
  monkeypatch.setattr(graph_visualizer, '__file__', str(repository / 'src' / 'scripts' / 'utils' / 'graph_visualizer.py'))
  _configure_main(monkeypatch, module)

  def algorithm(graph, start_node, end_node, *_args, epoch_callback, **_kwargs):
    pheromones = {0: np.array([1.0]), 1: np.array([])}
    for epoch in (1, 2):
      epoch_callback(
        {
          'epoch': epoch,
          'stage': 'pheromone_update',
          'pheromones': pheromones,
          'iteration_best_path': [start_node, end_node],
          'iteration_best_cost': 1.0,
          'global_best_path': [start_node, end_node],
          'global_best_cost': 1.0,
        }
      )
    return [start_node, end_node], 1.0, 0.01, 2

  monkeypatch.setattr(module, 'ACO', algorithm)
  monkeypatch.setattr(module, 'ACS', algorithm)
  monkeypatch.setattr(module, 'ABW', algorithm)

  module.main([])

  output_directory = repository / 'tmp'
  for name, title in (('aco', 'ACO pheromone evolution'), ('acs', 'ACS pheromone evolution'), ('bwas', 'BWAS pheromone evolution')):
    history_path = output_directory / f'{name}_pheromone_history.jsonl'
    html_path = output_directory / f'{name}_pheromone_animation.html'
    records = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()]
    assert [record['epoch'] for record in records[1:]] == [1, 2]
    assert html_path.exists()
    html = html_path.read_text(encoding='utf-8')
    assert title in html
    assert 'Dijkstra reference route' in html
    assert '17 ants' in html


def test_identical_endpoints_skip_algorithms_and_keep_zero_epoch_output(monkeypatch, capsys):
  import src.main as module

  class FakeWriter:
    def __init__(self, _graph, _name):
      pass

    def __call__(self, _observation):
      pytest.fail('trivial routes must not record algorithm epochs')

    def write_animation(self, **_kwargs):
      return None

  _configure_main(monkeypatch, module, writer_class=FakeWriter)
  monkeypatch.setattr(module, '_prompt_node', lambda *_: 0)
  monkeypatch.setattr(module, 'dijkstra', lambda *_: [0])
  monkeypatch.setattr(module, 'ACO', lambda *_args, **_kwargs: pytest.fail('ACO must not run'))
  monkeypatch.setattr(module, 'ACS', lambda *_args, **_kwargs: pytest.fail('ACS must not run'))
  monkeypatch.setattr(module, 'ABW', lambda *_args, **_kwargs: pytest.fail('ABW must not run'))

  module.main([])

  assert capsys.readouterr().out.splitlines() == [
    'ACO route solution: [0]',
    'ACO cost: 0.0',
    'ACO epochs: 0',
    'ACS route solution: [0]',
    'ACS cost: 0.0',
    'ACS epochs: 0',
    'ABW route solution: [0]',
    'ABW cost: 0.0',
    'ABW epochs: 0',
  ]


def test_bwas_uses_safe_defaults_when_optional_settings_are_absent(monkeypatch):
  import src.main as module

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

  class FakeWriter:
    def __init__(self, _graph, _name):
      pass

    def __call__(self, _observation):
      pass

    def write_animation(self, **_kwargs):
      return None

  _configure_main(monkeypatch, module, writer_class=FakeWriter)
  monkeypatch.setattr(module, '_resolved_settings', lambda _: minimal_settings)
  monkeypatch.setattr(module, 'ACO', lambda *_args, **_kwargs: ([0, 1], 1.0, 0.0, 1))
  monkeypatch.setattr(module, 'ACS', lambda *_args, **_kwargs: ([0, 1], 1.0, 0.0, 1))

  def fake_bwas(*_args, **kwargs):
    captured.update(kwargs)
    return [0, 1], 1.0, 0.0, 1

  monkeypatch.setattr(module, 'ABW', fake_bwas)

  module.main([])

  callback = captured.pop('epoch_callback')
  assert callable(callback)
  assert captured == {
    'worst_penalty_rate': None,
    'mutation_probability': 0.05,
    'mutation_scale': 2.0,
    'restart_stagnation': None,
    'min_pheromone_lvl': None,
    'global_best_patience': 2,
  }
