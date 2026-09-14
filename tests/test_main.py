import pytest

from src.scripts.main import prepare_routing_problem


def test_preflight_infers_aligned_walking_types_without_mutating_input():
  graph = {'node_index': {0, 1}, 'connections': {0: [1], 1: []}, 'weights': {0: [2.0], 1: []}}

  problem = prepare_routing_problem(graph, 0, 1)

  assert problem.graph['edge_types'] == {0: ['walk'], 1: []}
  assert 'edge_types' not in graph


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

  assert problem.graph is graph
  assert bus_node in problem.graph['node_index']
