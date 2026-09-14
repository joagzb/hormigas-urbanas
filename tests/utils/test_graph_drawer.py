import json

import numpy as np
import plotly.graph_objects as go
import pytest

from src.scripts.utils.graph_visualizer import (
  ExperimentOutputWriter,
  PheromoneHistoryWriter,
  build_graph_from_dict,
  build_pheromone_animation,
  calculate_animation_stride,
  draw_graph,
  draw_pheromone_history,
  load_pheromone_history,
  node_style,
  stable_edge_order,
)


@pytest.mark.parametrize(('total_iterations', 'expected_stride'), [(0, 1), (1, 1), (100, 1), (101, 2), (1000, 10), (1001, 10)])
def test_animation_stride_scales_with_completed_iterations(total_iterations, expected_stride):
  assert calculate_animation_stride(total_iterations) == expected_stride


def test_draw_graph_writes_interactive_html(tmp_path):
  graph = {'node_index': {0, 1, 2}, 'connections': {0: [1], 1: [2], 2: []}, 'weights': {0: [2.0], 1: [3.0], 2: []}}
  output_path = tmp_path / 'graph.html'

  figure = draw_graph(graph, path=[0, 1, 2], save_path=str(output_path))

  assert isinstance(figure, go.Figure)
  assert output_path.exists()
  assert 'plotly' in output_path.read_text(encoding='utf-8').lower()


def test_bus_node_style_comes_from_service_metadata():
  bus_node = 'bus:line:outbound:0'
  graph = {
    'node_index': {0, bus_node},
    'connections': {0: [bus_node], bus_node: [0]},
    'weights': {0: [1.4], bus_node: [0.01]},
    'edge_types': {0: ['board'], bus_node: ['alight']},
    'buses': [{'line_id': 'line', 'direction': 'outbound', 'stops': [(0, bus_node)]}],
  }

  graph_nx = build_graph_from_dict(graph)
  colors, sizes = node_style(graph_nx)
  style = {node: (color, size) for node, color, size in zip(graph_nx.nodes, colors, sizes)}

  assert style[bus_node] == ('orange', 450)
  assert style[0] == ('lightblue', 300)


def test_build_graph_rejects_misaligned_edge_types():
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}, 'edge_types': {0: [], 1: []}}

  with pytest.raises(ValueError, match='connections and edge_types are not aligned'):
    build_graph_from_dict(graph)


def test_visualizer_requires_edge_types_for_opaque_bus_nodes():
  bus_node = 'bus:line:outbound:0'
  graph = {'node_index': {0, bus_node}, 'connections': {0: [bus_node], bus_node: []}, 'weights': {0: [1.0], bus_node: []}}

  with pytest.raises(ValueError, match='edge_types'):
    stable_edge_order(graph)


def test_stable_edge_order_preserves_directed_parallel_edges_and_opaque_ids():
  bus_node = 'vehicle-X'
  graph = {
    'node_index': {0, bus_node},
    'connections': {bus_node: [0], 0: [bus_node, bus_node]},
    'weights': {bus_node: [0.1], 0: [1.0, 2.0]},
    'edge_types': {bus_node: ['alight'], 0: ['board', 'walk']},
  }

  edges = stable_edge_order(graph)

  assert [(edge.source, edge.adjacency_index, edge.target) for edge in edges] == [(0, 0, bus_node), (0, 1, bus_node), (bus_node, 0, 0)]
  assert [edge.edge_type for edge in edges] == ['board', 'walk', 'alight']


def test_history_writer_truncates_and_loader_keeps_first_sampled_and_final(tmp_path):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: [0]}, 'weights': {0: [1.0], 1: [1.0]}, 'edge_types': {0: ['walk'], 1: ['walk']}}
  history_path = tmp_path / 'history.jsonl'
  history_path.write_text('old content', encoding='utf-8')
  pheromones = {0: np.array([1.0]), 1: np.array([10.0])}
  writer = PheromoneHistoryWriter(graph, history_path)

  for epoch in range(1, 5):
    value = epoch
    pheromones[0][0] = value
    writer(
      {
        'epoch': epoch,
        'stage': 'pheromone_update',
        'pheromones': pheromones,
        'iteration_best_path': [0, 1],
        'iteration_best_cost': 1.0,
        'global_best_path': [0, 1],
        'global_best_cost': 1.0,
      }
    )

  snapshots = load_pheromone_history(graph, history_path, stride=2, max_frames=3)
  lines = history_path.read_text(encoding='utf-8').splitlines()

  assert json.loads(lines[0])['type'] == 'urban-ants-pheromone-history'
  assert len(lines) == 5
  assert [(snapshot['epoch'], snapshot['stage']) for snapshot in snapshots] == [(1, 'pheromone_update'), (2, 'pheromone_update'), (4, 'pheromone_update')]
  assert [snapshot['pheromones'][0] for snapshot in snapshots] == [1, 2, 4]


def test_history_round_trips_mixed_ids_parallel_edges_and_infinite_cost(tmp_path):
  bus_node = 'vehicle-X'
  graph = {
    'node_index': {0, bus_node},
    'connections': {bus_node: [0], 0: [bus_node, bus_node]},
    'weights': {bus_node: [0.1], 0: [1.0, 2.0]},
    'edge_types': {bus_node: ['alight'], 0: ['board', 'walk']},
  }
  history_path = tmp_path / 'history.jsonl'
  writer = PheromoneHistoryWriter(graph, history_path)

  writer(
    {
      'epoch': 1,
      'stage': 'pheromone_update',
      'pheromones': {0: np.array([1.0, 2.0]), bus_node: np.array([3.0])},
      'iteration_best_path': None,
      'iteration_best_cost': np.inf,
      'global_best_path': [0, bus_node, 0],
      'global_best_cost': 1.1,
    }
  )

  record = json.loads(history_path.read_text(encoding='utf-8').splitlines()[1])
  history_text = history_path.read_text(encoding='utf-8')
  snapshot = load_pheromone_history(graph, history_path)[0]

  assert record['iteration_best_cost'] is None
  assert 'Infinity' not in history_text
  assert 'NaN' not in history_text
  assert record['global_best_path'] == [0, 1, 0]
  assert snapshot['pheromones'] == pytest.approx([1.0, 2.0, 3.0])
  assert snapshot['global_best_path'] == [0, bus_node, 0]
  assert np.isinf(snapshot['iteration_best_cost'])


@pytest.mark.parametrize('invalid_value', [np.nan, np.inf, -np.inf])
def test_history_writer_rejects_non_finite_pheromones(tmp_path, invalid_value):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  history_path = tmp_path / 'history.jsonl'
  writer = PheromoneHistoryWriter(graph, history_path)

  with pytest.raises(ValueError, match='pheromones must contain finite numeric values'):
    writer({'epoch': 1, 'pheromones': {0: np.array([invalid_value]), 1: np.array([])}})

  assert len(history_path.read_text(encoding='utf-8').splitlines()) == 1


@pytest.mark.parametrize('invalid_value', [None, np.nan, np.inf, -np.inf])
def test_history_loader_rejects_non_finite_pheromones(tmp_path, invalid_value):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  history_path = tmp_path / 'history.jsonl'
  writer = PheromoneHistoryWriter(graph, history_path)
  writer({'epoch': 1, 'pheromones': {0: np.array([1.0]), 1: np.array([])}})
  lines = history_path.read_text(encoding='utf-8').splitlines()
  record = json.loads(lines[1])
  record['pheromones'][0] = invalid_value
  lines[1] = json.dumps(record)
  history_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

  with pytest.raises(ValueError, match='pheromones must contain finite numeric values'):
    load_pheromone_history(graph, history_path)


def test_history_preserves_null_optional_cost_and_path_fields(tmp_path):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  history_path = tmp_path / 'history.jsonl'
  writer = PheromoneHistoryWriter(graph, history_path)

  writer({'epoch': 1, 'pheromones': {0: np.array([1.0]), 1: np.array([])}})

  record = json.loads(history_path.read_text(encoding='utf-8').splitlines()[1])
  snapshot = load_pheromone_history(graph, history_path)[0]

  assert record['iteration_best_path'] is None
  assert record['iteration_best_cost'] is None
  assert record['global_best_path'] is None
  assert record['global_best_cost'] is None
  assert snapshot['iteration_best_path'] is None
  assert np.isinf(snapshot['iteration_best_cost'])
  assert snapshot['global_best_path'] is None
  assert np.isinf(snapshot['global_best_cost'])


@pytest.mark.parametrize('invalid_value', [np.nan, np.inf, -np.inf])
def test_animation_rejects_non_finite_pheromones(invalid_value):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  snapshot = {'epoch': 1, 'pheromones': np.array([invalid_value])}

  with pytest.raises(ValueError, match='pheromones must contain finite numeric values'):
    build_pheromone_animation(graph, [snapshot])


def test_history_loader_rejects_a_different_graph(tmp_path):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  history_path = tmp_path / 'history.jsonl'
  PheromoneHistoryWriter(graph, history_path)
  changed_graph = {**graph, 'weights': {0: [2.0], 1: []}}

  with pytest.raises(ValueError, match='graph alignment mismatch'):
    load_pheromone_history(changed_graph, history_path)


def test_draw_history_builds_minimal_route_frames_without_showing(tmp_path, monkeypatch):
  bus_node = 'bus:line:outbound:0'
  graph = {
    'node_index': {0, bus_node, 1},
    'connections': {0: [bus_node], bus_node: [1], 1: []},
    'weights': {0: [1.4], bus_node: [0.3], 1: []},
    'edge_types': {0: ['board'], bus_node: ['ride'], 1: []},
    'buses': [{'line_id': 'line', 'direction': 'outbound', 'stops': [(0, bus_node)]}],
  }
  history_path = tmp_path / 'history.jsonl'
  writer = PheromoneHistoryWriter(graph, history_path)
  for epoch in range(1, 3):
    if epoch == 1:
      global_best_path = [0, bus_node]
    else:
      global_best_path = [0, bus_node, 1]
    writer(
      {
        'epoch': epoch,
        'stage': 'pheromone_update',
        'pheromones': {0: np.array([epoch]), bus_node: np.array([2 * epoch]), 1: np.array([])},
        'iteration_best_path': [0, bus_node, 1],
        'iteration_best_cost': 1.7,
        'global_best_path': global_best_path,
        'global_best_cost': 1.7,
      }
    )

  monkeypatch.setattr(go.Figure, 'show', lambda self: pytest.fail('draw_pheromone_history displayed the figure inline'))

  figure = draw_pheromone_history(graph, history_path, reference_path=[0, bus_node, 1], title='Hidden algorithm details', show=False)

  assert len(figure.frames) == 2
  assert len(figure.layout.updatemenus[0].buttons) == 2
  assert [step.label for step in figure.layout.sliders[0].steps] == ['Iteration 1', 'Iteration 2']
  assert [frame.name for frame in figure.frames] == ['1', '2']
  assert [frame.layout.title.text for frame in figure.frames] == ['Iteration 1', 'Iteration 2']
  assert figure.layout.title.text == 'Iteration 1'
  assert [trace.name for trace in figure.frames[0].data] == ['Structural graph edges', 'Dijkstra reference route', 'Global best-found route', 'Nodes']
  structural_edges = figure.frames[0].data[0]
  assert structural_edges.mode == 'lines'
  assert structural_edges.showlegend is False
  assert structural_edges.text is None and structural_edges.customdata is None
  assert all(trace.hoverinfo == 'skip' for trace in figure.frames[0].data)
  assert all(trace.hovertext is None and trace.hovertemplate is None for trace in figure.frames[0].data)
  assert figure.frames[0].data[0].x == figure.frames[1].data[0].x
  assert figure.frames[0].data[2].x != figure.frames[1].data[2].x
  serialized = figure.to_json()
  assert bus_node in serialized
  assert 'Iteration best-found route' not in serialized
  assert 'pheromone_update' not in serialized
  assert 'pheromones restarted' not in serialized
  assert 'Hidden algorithm details' not in serialized
  assert 'Type: board' not in serialized
  assert 'Type: ride' not in serialized
  assert 'Bus line:' not in serialized


def test_draw_history_writes_interactive_html_to_created_output_directory(tmp_path):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  output_directory = tmp_path / 'tmp'
  output_directory.mkdir(parents=True, exist_ok=True)
  history_path = output_directory / 'aco_pheromone_history.jsonl'
  html_path = output_directory / 'aco_pheromone_animation.html'
  writer = PheromoneHistoryWriter(graph, history_path)
  writer({'epoch': 1, 'pheromones': {0: np.array([1.0]), 1: np.array([])}})

  figure = draw_pheromone_history(graph, history_path, show=False)
  figure.write_html(html_path)

  assert html_path.exists()
  assert 'plotly' in html_path.read_text(encoding='utf-8').lower()


def test_experiment_output_writer_owns_cwd_tmp_paths_and_html(monkeypatch, tmp_path):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  monkeypatch.chdir(tmp_path)
  output = ExperimentOutputWriter(graph, 'aco')
  output({'epoch': 1, 'pheromones': {0: np.array([1.0]), 1: np.array([])}})

  html_path = output.write_animation()

  assert output.history_path == tmp_path / 'tmp' / 'aco_pheromone_history.jsonl'
  assert html_path == tmp_path / 'tmp' / 'aco_pheromone_animation.html'
  assert html_path.exists()


def test_experiment_output_writer_uses_adaptive_stride(monkeypatch, tmp_path):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  monkeypatch.chdir(tmp_path)
  output = ExperimentOutputWriter(graph, 'aco')
  for epoch in range(1, 102):
    output({'epoch': epoch, 'pheromones': {0: np.array([1.0]), 1: np.array([])}})
  captured = {}

  def fake_draw_pheromone_history(*args, **kwargs):
    captured.update(kwargs)
    return go.Figure()

  monkeypatch.setattr('src.scripts.utils.graph_visualizer.draw_pheromone_history', fake_draw_pheromone_history)

  output.write_animation()

  assert captured['stride'] == 2
  assert captured['max_frames'] == 100
  assert captured['show'] is False


def test_adaptive_stride_retains_final_iteration(tmp_path):
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [1.0], 1: []}}
  history_path = tmp_path / 'history.jsonl'
  writer = PheromoneHistoryWriter(graph, history_path)
  for epoch in range(1, 102):
    writer({'epoch': epoch, 'pheromones': {0: np.array([1.0]), 1: np.array([])}})

  snapshots = load_pheromone_history(graph, history_path, stride=calculate_animation_stride(101))

  assert snapshots[0]['epoch'] == 1
  assert snapshots[-1]['epoch'] == 101
  assert len(snapshots) == 52


def test_animation_handles_no_recorded_epochs_and_no_route():
  graph = {'node_index': {'start'}, 'connections': {'start': []}, 'weights': {'start': []}}

  figure = build_pheromone_animation(graph, [], reference_path=None)

  assert not figure.frames
  assert figure.layout.title.text == 'Iteration 0'
  assert [trace.name for trace in figure.data] == ['Structural graph edges', 'Dijkstra reference route', 'Global best-found route', 'Nodes']
