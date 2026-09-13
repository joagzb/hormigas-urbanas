import copy

import numpy as np
import pytest

from src.scripts.utils.generators import deterministic_route_cost, merge_bus_and_map_graph, generate_pheromone_map, validate_graph
from src.scripts.utils.route_finder import dijkstra


def _map_graph():
  return {
    'node_index': {0, 1, 2},
    'connections': {0: [1], 1: [0, 2], 2: [1]},
    'weights': {0: [10.0], 1: [10.0, 10.0], 2: [10.0]},
    'edge_types': {0: ['walk'], 1: ['walk', 'walk'], 2: ['walk']},
  }


def _bus_service(line_id='0', direction='outbound', weight=3.0):
  nodes = [f'bus:{line_id}:{direction}:{i}' for i in range(3)]
  return {
    'name': f'bus line {line_id} {direction}',
    'line_id': line_id,
    'direction': direction,
    'route': [0, 1, 2],
    'node_bus_index': set(nodes),
    'connections': {nodes[0]: [nodes[1]], nodes[1]: [nodes[2]], nodes[2]: []},
    'weights': {nodes[0]: [weight], nodes[1]: [weight], nodes[2]: []},
    'edge_types': {nodes[0]: ['ride'], nodes[1]: ['ride'], nodes[2]: []},
    'stops': list(zip([0, 1, 2], nodes)),
  }


def test_merge_bus_and_map_graph_adds_aligned_transfer_edges(monkeypatch):
  """Merging adds get-on edges at non-terminal stops and get-off edges everywhere."""
  map_graph = _map_graph()
  buses_graph = [_bus_service()]

  # patch boarding/alighting costs
  import src.scripts.utils.generators as mod

  monkeypatch.setattr(mod, 'calculate_bus_get_on_cost', lambda: 3.3)
  monkeypatch.setattr(mod, 'calculate_bus_get_off_cost', lambda: 0.01)

  merged = merge_bus_and_map_graph(map_graph, buses_graph)

  # Bus nodes are merged into node_index
  bus_nodes = [f'bus:0:outbound:{i}' for i in range(3)]
  assert set(bus_nodes).issubset(merged['node_index'])

  # Get-on edges from map node to bus node for all but last stop of route
  assert merged['connections'][0][-1] == bus_nodes[0]
  assert merged['weights'][0][-1] == 3.3
  assert merged['edge_types'][0][-1] == 'board'
  assert merged['connections'][1][-1] == bus_nodes[1]
  assert merged['weights'][1][-1] == 3.3

  # Get-off edges from bus node to map node for every stop
  assert merged['connections'][bus_nodes[0]][-1] == 0
  assert merged['weights'][bus_nodes[0]][-1] == 0.01
  assert merged['edge_types'][bus_nodes[0]][-1] == 'alight'
  assert merged['connections'][bus_nodes[1]][-1] == 1
  assert merged['connections'][bus_nodes[2]][-1] == 2
  assert all(len(merged['connections'][node]) == len(merged['weights'][node]) == len(merged['edge_types'][node]) for node in merged['node_index'])

  # buses list preserved
  assert merged['buses'] == buses_graph


def test_merge_is_idempotent_independent_and_does_not_alias_inputs():
  map_graph = _map_graph()
  buses_graph = [_bus_service()]
  original_map = copy.deepcopy(map_graph)
  original_buses = copy.deepcopy(buses_graph)

  merged = merge_bus_and_map_graph(map_graph, buses_graph)
  repeated = merge_bus_and_map_graph(merged, buses_graph)

  assert map_graph == original_map
  assert buses_graph == original_buses
  assert repeated == merged
  merged['connections'][0].append('changed')
  merged['buses'][0]['route'].append(99)
  assert map_graph == original_map
  assert buses_graph == original_buses


def test_merge_treats_missing_map_edge_types_as_walk():
  map_graph = _map_graph()
  map_graph.pop('edge_types')

  merged = merge_bus_and_map_graph(map_graph, [_bus_service()])

  assert merged['edge_types'][0][0] == 'walk'
  assert merged['edge_types'][1][:2] == ['walk', 'walk']
  assert 'edge_types' not in map_graph


def test_merge_rejects_missing_edge_types_from_an_existing_multimodal_graph():
  multimodal_graph = merge_bus_and_map_graph(_map_graph(), [_bus_service()])
  multimodal_graph.pop('edge_types')

  with pytest.raises(ValueError, match='edge_types'):
    merge_bus_and_map_graph(multimodal_graph, [])


def test_merge_rejects_tagged_bus_nodes_without_edge_types():
  bus_node = 'bus:line:outbound:0'
  malformed_graph = {'node_index': {bus_node}, 'connections': {bus_node: []}, 'weights': {bus_node: []}, 'buses': []}

  with pytest.raises(ValueError, match='edge_types'):
    merge_bus_and_map_graph(malformed_graph, [])


def test_merge_rejects_conflicting_services_collisions_and_invalid_alignment():
  service = _bus_service()
  merged = merge_bus_and_map_graph(_map_graph(), [service])
  conflicting = _bus_service(weight=4.0)
  with pytest.raises(ValueError, match='Conflicting bus service'):
    merge_bus_and_map_graph(merged, [conflicting])

  collision = _bus_service(line_id='other')
  first_node = next(iter(collision['node_bus_index']))
  collision['node_bus_index'].remove(first_node)
  collision['node_bus_index'].add(0)
  collision['connections'][0] = collision['connections'].pop(first_node)
  collision['weights'][0] = collision['weights'].pop(first_node)
  collision['edge_types'][0] = collision['edge_types'].pop(first_node)
  for node, neighbors in collision['connections'].items():
    collision['connections'][node] = [0 if neighbor == first_node else neighbor for neighbor in neighbors]
  collision['stops'] = [(map_node, 0 if bus_node == first_node else bus_node) for map_node, bus_node in collision['stops']]
  with pytest.raises(ValueError, match='collide'):
    merge_bus_and_map_graph(_map_graph(), [collision])

  invalid_map = _map_graph()
  invalid_map['edge_types'][0] = []
  with pytest.raises(ValueError, match='edge_types'):
    merge_bus_and_map_graph(invalid_map, [])

  invalid_weight = _map_graph()
  invalid_weight['weights'][0][0] = -1
  with pytest.raises(ValueError, match='invalid weight'):
    merge_bus_and_map_graph(invalid_weight, [])

  unknown_neighbor = _map_graph()
  unknown_neighbor['connections'][0][0] = 99
  with pytest.raises(ValueError, match='unknown node'):
    merge_bus_and_map_graph(unknown_neighbor, [])

  missing_bus_edge_types = _bus_service(line_id='missing-types')
  missing_bus_edge_types.pop('edge_types')
  with pytest.raises(ValueError, match='edge_types'):
    merge_bus_and_map_graph(_map_graph(), [missing_bus_edge_types])


def test_validate_graph_rejects_unknown_edge_type():
  graph = _map_graph()
  graph['edge_types'][0][0] = 'teleport'

  with pytest.raises(ValueError, match='unknown edge type'):
    validate_graph(graph)


def test_transfer_route_contains_two_board_edges():
  first = _bus_service(line_id='first')
  first['route'] = [0, 1]
  first['stops'] = first['stops'][:2]
  terminal = 'bus:first:outbound:1'
  first['connections'][terminal] = []
  first['weights'][terminal] = []
  first['edge_types'][terminal] = []
  unused = 'bus:first:outbound:2'
  first['node_bus_index'].remove(unused)
  first['connections'].pop(unused)
  first['weights'].pop(unused)
  first['edge_types'].pop(unused)

  second = _bus_service(line_id='second')
  second['route'] = [1, 2]
  second['stops'] = [(1, 'bus:second:outbound:0'), (2, 'bus:second:outbound:1')]
  terminal = 'bus:second:outbound:1'
  second['connections'][terminal] = []
  second['weights'][terminal] = []
  second['edge_types'][terminal] = []
  unused = 'bus:second:outbound:2'
  second['node_bus_index'].remove(unused)
  second['connections'].pop(unused)
  second['weights'].pop(unused)
  second['edge_types'].pop(unused)

  merged = merge_bus_and_map_graph(_map_graph(), [first, second])
  path = dijkstra(merged, 0, 2)
  path_edge_types = [merged['edge_types'][start][merged['connections'][start].index(end)] for start, end in zip(path, path[1:])]

  assert path_edge_types.count('board') == 2
  assert path_edge_types.count('alight') == 2


def test_generate_pheromone_map_aligns_with_connections():
  """Pheromone map mirrors connections and initializes with the given level."""
  map_graph = {'connections': {0: [1, 2], 1: [2], 2: []}}
  initial = 0.5
  pher = generate_pheromone_map(map_graph, initial_lvl=initial)

  # same keys and lengths as connections; values start at initial
  assert set(pher.keys()) == {0, 1, 2}
  assert pher[0].shape == (2,)
  assert pher[1].shape == (1,)
  assert pher[2].shape == (0,)
  assert np.allclose(pher[0], initial)
  assert np.allclose(pher[1], initial)


def test_deterministic_route_cost_backtracks_from_cycles_and_uses_aligned_weights():
  graph = {'node_index': {0, 1, 2, 3}, 'connections': {0: [2, 1], 1: [0], 2: [3], 3: []}, 'weights': {0: [4.0, 1.0], 1: [1.0], 2: [2.0], 3: []}}

  assert deterministic_route_cost(graph, 0, 3) == 6.0


def test_deterministic_route_cost_handles_trivial_and_unreachable_routes():
  graph = {'node_index': {0, 1, 2}, 'connections': {0: [1], 1: [0], 2: []}, 'weights': {0: [1.0], 1: [1.0], 2: []}}

  assert deterministic_route_cost(graph, 0, 0) == 0.0
  assert np.isinf(deterministic_route_cost(graph, 0, 2))
