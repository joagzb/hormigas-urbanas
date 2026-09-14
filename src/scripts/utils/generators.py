import copy

import numpy as np
from .weights import calculate_bus_get_off_cost, calculate_bus_get_on_cost, calculate_bus_time_travel_cost

EDGE_TYPES = {'walk', 'board', 'ride', 'alight'}


def validate_graph(graph, require_edge_types=True):
  """Validate an aligned dictionary graph without mutating it.

  ``graph`` must contain the same opaque, hashable node IDs in ``node_index``,
  ``connections``, and ``weights``. Neighbor, weight, and optional edge-type
  rows must stay positionally aligned. The function returns ``None`` and raises
  ``ValueError`` for malformed nodes, weights, edge types, or row alignment.
  """
  nodes = set(graph.get('node_index', []))
  connections = graph.get('connections', {})
  weights = graph.get('weights', {})
  edge_types = graph.get('edge_types')

  if set(connections) != nodes or set(weights) != nodes:
    raise ValueError('Graph nodes, connections, and weights must have the same keys')
  if require_edge_types and edge_types is None:
    raise ValueError('Graph nodes and edge_types must have the same keys')
  if edge_types is not None and set(edge_types) != nodes:
    raise ValueError('Graph nodes and edge_types must have the same keys')

  for node in nodes:
    node_connections = connections[node]
    node_weights = weights[node]
    if edge_types is None:
      node_edge_types = None
    else:
      node_edge_types = edge_types[node]
    if len(node_connections) != len(node_weights):
      raise ValueError(f'Graph connections and weights are not aligned at node {node!r}')
    if node_edge_types is not None and len(node_connections) != len(node_edge_types):
      raise ValueError(f'Graph connections and edge_types are not aligned at node {node!r}')
    for neighbor, weight in zip(node_connections, node_weights):
      if neighbor not in nodes:
        raise ValueError(f'Graph edge from {node!r} references unknown node {neighbor!r}')
      try:
        valid_weight = np.isfinite(weight) and weight >= 0
      except TypeError:
        valid_weight = False
      if not valid_weight:
        raise ValueError(f'Graph edge from {node!r} has an invalid weight')
    if node_edge_types is not None and any(edge_type not in EDGE_TYPES for edge_type in node_edge_types):
      raise ValueError(f'Graph edge from {node!r} has an unknown edge type')


def _service_identity(bus_graph):
  try:
    return bus_graph['line_id'], bus_graph['direction']
  except KeyError as error:
    raise ValueError('Bus services require line_id and direction') from error


def merge_bus_and_map_graph(map_graph, buses_graph):
  """Merge bus services into an independent copy of the map graph.

  ``map_graph`` and each bus service use positionally aligned adjacency, weight,
  and edge-type rows with opaque node IDs. The returned graph is independent
  from both inputs. Missing edge types on a walking-only map are inferred as
  ``walk``; transfer edges and bus service metadata are added to the copy.
  Services are identified by ``(line_id, direction)`` so merging an identical
  service again is a no-op. Invalid or colliding services raise ``ValueError``.
  """
  merged_graph = copy.deepcopy(map_graph)
  has_tagged_bus_nodes = any(isinstance(node, str) and node.startswith('bus:') for node in merged_graph.get('node_index', []))
  if 'edge_types' not in merged_graph and not merged_graph.get('buses') and not has_tagged_bus_nodes:
    merged_graph['edge_types'] = {node: ['walk'] * len(connections) for node, connections in merged_graph['connections'].items()}
  validate_graph(merged_graph)
  merged_graph.setdefault('buses', [])
  existing_services = {}
  for bus_graph in merged_graph['buses']:
    identity = _service_identity(bus_graph)
    if identity in existing_services and existing_services[identity] != bus_graph:
      raise ValueError(f'Conflicting bus service {identity!r}')
    existing_services[identity] = bus_graph

  for source_bus_graph in buses_graph:
    bus_graph = copy.deepcopy(source_bus_graph)
    validate_graph(
      {
        'node_index': bus_graph.get('node_bus_index', set()),
        'connections': bus_graph.get('connections', {}),
        'weights': bus_graph.get('weights', {}),
        'edge_types': bus_graph.get('edge_types', {}),
      }
    )
    identity = _service_identity(bus_graph)
    if identity in existing_services:
      if existing_services[identity] == bus_graph:
        continue
      raise ValueError(f'Conflicting bus service {identity!r}')

    bus_nodes = set(bus_graph['node_bus_index'])
    collisions = bus_nodes.intersection(merged_graph['node_index'])
    if collisions:
      raise ValueError(f'Bus node IDs collide with existing nodes: {collisions!r}')

    route = bus_graph['route']
    stops = bus_graph['stops']
    if len(route) != len(stops):
      raise ValueError('Bus route and stops must be aligned')
    if [map_node for map_node, _ in stops] != route:
      raise ValueError('Bus route and stop map nodes must be aligned')
    if any(bus_node not in bus_nodes for _, bus_node in stops):
      raise ValueError('Bus stops must reference service bus nodes')
    if any(map_node not in merged_graph['node_index'] for map_node, _ in stops):
      raise ValueError('Bus stops must reference existing map nodes')

    # Add the validated service rows before connecting it to the street map.
    merged_graph['node_index'].update(bus_nodes)
    merged_graph['connections'].update(copy.deepcopy(bus_graph['connections']))
    merged_graph['weights'].update(copy.deepcopy(bus_graph['weights']))
    merged_graph['edge_types'].update(copy.deepcopy(bus_graph['edge_types']))

    for i, (start_map_node, start_bus_node) in enumerate(stops):
      if i < len(route) - 1:
        cost_get_on = calculate_bus_get_on_cost()
        merged_graph['connections'][start_map_node].append(start_bus_node)
        merged_graph['weights'][start_map_node].append(cost_get_on)
        merged_graph['edge_types'][start_map_node].append('board')

      cost_get_off = calculate_bus_get_off_cost()
      merged_graph['connections'][start_bus_node].append(start_map_node)
      merged_graph['weights'][start_bus_node].append(cost_get_off)
      merged_graph['edge_types'][start_bus_node].append('alight')

    merged_graph['buses'].append(copy.deepcopy(bus_graph))
    existing_services[identity] = merged_graph['buses'][-1]

  validate_graph(merged_graph)
  return merged_graph


def generate_pheromone_map(map_graph, initial_lvl):
  """Return new pheromone rows aligned with every graph adjacency row.

  Opaque node IDs are preserved as dictionary keys. Each returned NumPy row
  has the same length and order as ``map_graph['connections'][node]`` and is
  filled with ``initial_lvl``. The input graph is not mutated.
  """

  pheromone_path = {}
  for node, connections in map_graph['connections'].items():
    pheromone_path[node] = initial_lvl + np.zeros(len(connections))

  return pheromone_path


def deterministic_route_cost(graph_map, start_node, end_node):
  """Return a deterministic first-found route cost for tau0 initialization.

  The graph uses opaque node IDs and positionally aligned connection/weight
  rows. A lowest-edge-cost-first depth-first search returns the sum of original
  edge weights for its first complete route, ``0.0`` for identical endpoints,
  or ``np.inf`` when no route exists. It does not mutate the graph, call a
  shortest-path oracle, validate experiment results, or alter ant route costs.
  """
  if start_node == end_node:
    return 0.0

  connections = graph_map['connections']
  weights = graph_map['weights']
  nodes = graph_map['node_index']
  visited = {start_node}
  path_costs = []

  def ordered_edges(node):
    node_connections = connections.get(node, [])
    node_weights = weights.get(node, [])
    if len(node_connections) != len(node_weights):
      raise ValueError('Graph connections and weights must be aligned')
    return iter(sorted(zip(node_connections, node_weights), key=lambda edge: edge[1]))

  search_stack = [ordered_edges(start_node)]
  while search_stack:
    edges = search_stack[-1]
    try:
      next_node, edge_cost = next(edges)
    except StopIteration:
      search_stack.pop()
      if path_costs:
        path_costs.pop()
      continue

    if next_node not in nodes or next_node in visited:
      continue

    visited.add(next_node)
    path_costs.append(edge_cost)
    if next_node == end_node:
      cost = sum(path_costs)
      break
    search_stack.append(ordered_edges(next_node))
  else:
    return np.inf

  if cost < 0 or not np.isfinite(cost):
    raise ValueError('Baseline route cost must be finite and non-negative')
  return cost


def generate_square_city_graph(size, fixed_weight):
  """Return a bidirectional square walking graph with aligned edge metadata.

  ``size`` is the number of rows and columns and ``fixed_weight`` is assigned
  to every directed walking edge. Integer map IDs span ``0`` through
  ``size * size - 1``. The returned dictionary owns independent, positionally
  aligned ``connections``, ``weights``, and ``edge_types`` rows and has no file,
  display, random-state, or input-mutation side effects.
  """
  graph = {'node_index': set(range(size * size)), 'connections': [], 'weights': [], 'edge_types': []}

  for row in range(size):
    for column in range(size):
      current_node = row * size + column
      graph['connections'].append((current_node, []))
      graph['weights'].append((current_node, []))
      graph['edge_types'].append((current_node, []))

      for neighbor, available in ((current_node - size, row > 0), (current_node + size, row < size - 1), (current_node - 1, column > 0), (current_node + 1, column < size - 1)):
        if available:
          graph['connections'][-1][1].append(neighbor)
          graph['weights'][-1][1].append(fixed_weight)
          graph['edge_types'][-1][1].append('walk')

  graph['connections'] = dict(graph['connections'])
  graph['weights'] = dict(graph['weights'])
  graph['edge_types'] = dict(graph['edge_types'])
  return graph


def generate_bus_line_square_city(size, fixed_weight, line_id='UNIQUE', route=None):
  """Return outbound and inbound directed bus services for a square city.

  ``size`` defines the compatible square map, ``fixed_weight`` is converted to
  the configured bus travel cost, ``line_id`` distinguishes services, and an
  optional ``route`` supplies ordered physical map stops. Each occurrence gets
  an opaque string bus-node ID, so repeated physical stops remain collision
  free. Returned connection, weight, and ``ride`` type rows are positionally
  aligned. Inputs are not mutated and no files, prompts, or random state are
  used.
  """
  distance = calculate_bus_time_travel_cost(fixed_weight)
  if route is None:
    physical_route = list(range(min(5, size - 1), size * size, size))
  else:
    physical_route = list(route)

  buses = []
  directions = (('outbound', physical_route), ('inbound', list(reversed(physical_route))))
  for direction, directed_route in directions:
    bus_nodes = [f'bus:{line_id}:{direction}:{index}' for index in range(len(directed_route))]
    bus_graph = {
      'name': f'bus line {line_id} {direction}',
      'line_id': line_id,
      'direction': direction,
      'stops': list(zip(directed_route, bus_nodes)),
      'route': list(directed_route),
      'node_bus_index': set(bus_nodes),
      'connections': {},
      'weights': {},
      'edge_types': {},
    }
    for index, bus_node in enumerate(bus_nodes):
      has_next_stop = index < len(bus_nodes) - 1
      if has_next_stop:
        bus_graph['connections'][bus_node] = [bus_nodes[index + 1]]
        bus_graph['weights'][bus_node] = [distance]
        bus_graph['edge_types'][bus_node] = ['ride']
      else:
        bus_graph['connections'][bus_node] = []
        bus_graph['weights'][bus_node] = []
        bus_graph['edge_types'][bus_node] = []
    buses.append(bus_graph)
  return buses
