import builtins
import runpy
import sys

import pytest

from src.scripts.utils.dijkstra import dijkstra
from src.scripts.utils.generators import merge_bus_and_map_graph
from src.scripts.utils.toy_city_generators import _compute_route_cost, _route_recommendation, generate_bus_line_square_city, generate_square_city_graph


def test_legacy_module_reexports_generators_from_central_utility_module():
  assert generate_square_city_graph.__module__ == 'src.scripts.utils.generators'
  assert generate_bus_line_square_city.__module__ == 'src.scripts.utils.generators'


def test_route_recommendation_compares_numeric_costs_and_handles_missing_routes():
  assert _route_recommendation(3.0, 4.0) == "you'd better take the bus instead of walking."
  assert _route_recommendation(4.0, 4.0) == "you'd better go by foot."
  assert _route_recommendation(float('inf'), 4.0) == "you'd better go by foot."
  assert _route_recommendation(float('inf'), float('inf')) == 'no route could be found.'
  assert _compute_route_cost({}, None) == float('inf')


def test_bus_ids_are_deterministic_collision_free_and_include_both_directions():
  buses = generate_bus_line_square_city(10, 1, line_id='square')

  assert [(bus['line_id'], bus['direction']) for bus in buses] == [('square', 'outbound'), ('square', 'inbound')]
  all_nodes = [node for bus in buses for node in bus['node_bus_index']]
  assert len(all_nodes) == len(set(all_nodes))
  assert all(node.startswith('bus:square:') for node in all_nodes)
  assert buses[1]['route'] == list(reversed(buses[0]['route']))
  assert buses == generate_bus_line_square_city(10, 1, line_id='square')


def test_lines_can_share_and_repeat_physical_stops_without_bus_node_collisions():
  first = generate_bus_line_square_city(3, 1, line_id='first', route=[0, 1, 0])
  second = generate_bus_line_square_city(3, 1, line_id='second', route=[0, 1, 0])
  nodes = [node for bus in first + second for node in bus['node_bus_index']]

  assert len(nodes) == len(set(nodes))
  assert first[0]['stops'][0][0] == first[0]['stops'][2][0] == 0
  assert first[0]['stops'][0][1] != first[0]['stops'][2][1]


def test_nearby_route_walks_while_longer_route_uses_bus_in_both_directions():
  map_graph = generate_square_city_graph(10, 1)
  full_graph = merge_bus_and_map_graph(map_graph, generate_bus_line_square_city(10, 1))

  nearby = dijkstra(full_graph, 5, 15)
  outbound = dijkstra(full_graph, 5, 95)
  inbound = dijkstra(full_graph, 95, 5)

  assert nearby == [5, 15]
  assert any(isinstance(node, str) for node in outbound)
  assert any(isinstance(node, str) for node in inbound)
  assert _compute_route_cost(full_graph, outbound) < _compute_route_cost(map_graph, dijkstra(map_graph, 5, 95))


@pytest.mark.parametrize(('module_name', 'from_src'), [('src.scripts.utils.toy_city_generators', False), ('scripts.utils.toy_city_generators', True)])
def test_documented_toy_city_module_forms_run(monkeypatch, module_name, from_src):
  if from_src:
    monkeypatch.syspath_prepend('src')
  monkeypatch.setattr(builtins, 'input', lambda _: '')
  graph_visualizer = __import__(f'{module_name.rsplit(".", 1)[0]}.graph_visualizer', fromlist=['draw_graph'])
  monkeypatch.setattr(graph_visualizer, 'draw_graph', lambda *args, **kwargs: None)
  sys.modules.pop(module_name, None)

  runpy.run_module(module_name, run_name='__main__')
