from src.scripts.utils.dijkstra import dijkstra


def test_dijkstra_finds_path_on_linear_chain():
  """Finds the only path on a simple 0->1->2->3 chain."""
  simple_graph = {'node_index': {0, 1, 2, 3}, 'connections': {0: [1], 1: [2], 2: [3], 3: []}, 'weights': {0: [1.0], 1: [1.0], 2: [1.0], 3: []}}

  path = dijkstra(simple_graph, 0, 3)
  expected_path = [0, 1, 2, 3]
  assert path == expected_path


def test_dijkstra_handles_equal_distance_mixed_node_ids():
  graph = {
    'node_index': {0, 1, 'bus:line:outbound:0', 'end'},
    'connections': {0: [1, 'bus:line:outbound:0'], 1: ['end'], 'bus:line:outbound:0': ['end'], 'end': []},
    'weights': {0: [1.0, 1.0], 1: [1.0], 'bus:line:outbound:0': [1.0], 'end': []},
  }

  path = dijkstra(graph, 0, 'end')

  assert path in ([0, 1, 'end'], [0, 'bus:line:outbound:0', 'end'])
