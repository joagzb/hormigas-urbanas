"""Public experiment-boundary helpers for routing algorithms."""

import copy
from dataclasses import dataclass

from .utils.generators import validate_graph


def _requires_edge_types(graph):
  return bool(graph.get('buses')) or any(isinstance(node, str) and node.startswith('bus:') for node in graph.get('node_index', []))


@dataclass(frozen=True)
class RoutingProblem:
  """A validated graph and endpoint pair ready for algorithm execution.

  ``graph`` contains opaque node IDs and aligned connection, weight, and edge
  type rows. ``run`` returns a zero-epoch route for identical endpoints or
  delegates to the supplied colony function. The wrapper does not inspect,
  replace, or recompute ant-reported route costs.
  """

  graph: dict
  start_node: object
  end_node: object

  def run(self, algorithm, *args, **kwargs):
    """Run ``algorithm`` or return the boundary-owned trivial-route result."""
    if self.start_node == self.end_node:
      return [self.start_node], 0.0, 0.0, 0
    return algorithm(self.graph, self.start_node, self.end_node, *args, **kwargs)


def prepare_routing_problem(graph, start_node, end_node):
  """Validate and normalize public graph inputs at the experiment boundary.

  ``graph`` must use opaque, hashable IDs and aligned adjacency/weight rows.
  Walking-only graphs without ``edge_types`` are copied and receive aligned
  ``walk`` rows; multimodal graphs must provide explicit edge types. Both
  endpoints must exist. The returned ``RoutingProblem`` may own a shallow graph
  copy only when edge types are inferred; the supplied graph is never mutated.
  """
  prepared_graph = graph
  if 'edge_types' not in graph:
    if _requires_edge_types(graph):
      raise ValueError('Multimodal graphs require explicit edge_types')
    prepared_graph = copy.copy(graph)
    prepared_graph['edge_types'] = {node: ['walk'] * len(neighbors) for node, neighbors in graph.get('connections', {}).items()}

  validate_graph(prepared_graph)
  nodes = prepared_graph['node_index']
  if start_node not in nodes or end_node not in nodes:
    raise ValueError('start_node and end_node must exist in the graph')
  return RoutingProblem(prepared_graph, start_node, end_node)
