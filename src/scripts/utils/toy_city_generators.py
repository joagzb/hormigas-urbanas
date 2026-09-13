"""Utility generators for simple square graphs used in tests.

These helpers are primarily intended for quick experimentation and unit tests.
They build a deterministic square grid and a vertical bus line in both directions to
validate routing algorithms.
"""

from ..utils.graph_visualizer import draw_graph
from ..utils.weights import calculate_bus_time_travel_cost
from ..utils.generators import merge_bus_and_map_graph
from ..utils.route_finder import dijkstra


def generate_square_city_graph(size, fixed_weight):
  """
  Generate a square city graph with the given size and fixed edge weight.

  Parameters:
      size (int): The size of the city (number of nodes per side).
      fixed_weight (float): The weight for each edge.

  Returns:
      dict: A dictionary representing the graph with nodes, connections, and weights.
  """
  graph = {'node_index': set(range(size * size)), 'connections': [], 'weights': [], 'edge_types': []}

  for i in range(size):
    for j in range(size):
      current_node = i * size + j

      graph['connections'].append((current_node, []))
      graph['weights'].append((current_node, []))
      graph['edge_types'].append((current_node, []))

      if i > 0:  # North
        graph['connections'][-1][1].append(current_node - size)
        graph['weights'][-1][1].append(fixed_weight)
        graph['edge_types'][-1][1].append('walk')

      if i < size - 1:  # South
        graph['connections'][-1][1].append(current_node + size)
        graph['weights'][-1][1].append(fixed_weight)
        graph['edge_types'][-1][1].append('walk')

      if j > 0:  # West
        graph['connections'][-1][1].append(current_node - 1)
        graph['weights'][-1][1].append(fixed_weight)
        graph['edge_types'][-1][1].append('walk')

      if j < size - 1:  # East
        graph['connections'][-1][1].append(current_node + 1)
        graph['weights'][-1][1].append(fixed_weight)
        graph['edge_types'][-1][1].append('walk')

  graph['connections'] = dict(graph['connections'])
  graph['weights'] = dict(graph['weights'])
  graph['edge_types'] = dict(graph['edge_types'])

  return graph


def generate_bus_line_square_city(size, fixed_weight, line_id='UNIQUE', route=None):
  """
  Generate a bus line graph in a square city with the given size and fixed edge weight.

  Parameters:
      size (int): The size of the city (number of nodes per side).
      fixed_weight (float): The weight for each edge.

  Returns:
      list: The outbound and inbound directed bus services.
  """
  distance = calculate_bus_time_travel_cost(fixed_weight)
  if route is None:
    route = list(range(min(5, size - 1), size * size, size))

  buses = []
  directions = (('outbound', list(route)), ('inbound', list(reversed(route))))
  for direction, directed_route in directions:
    bus_nodes = [f'bus:{line_id}:{direction}:{i}' for i in range(len(directed_route))]
    bus_dict = {
      'name': f'bus line {line_id} {direction}',
      'line_id': line_id,
      'direction': direction,
      'stops': list(zip(directed_route, bus_nodes)),
      'route': directed_route,
      'node_bus_index': set(bus_nodes),
      'connections': {},
      'weights': {},
      'edge_types': {},
    }

    for i, bus_current in enumerate(bus_nodes):
      if i < len(bus_nodes) - 1:
        bus_dict['connections'][bus_current] = [bus_nodes[i + 1]]
        bus_dict['weights'][bus_current] = [distance]
        bus_dict['edge_types'][bus_current] = ['ride']
      else:
        bus_dict['connections'][bus_current] = []
        bus_dict['weights'][bus_current] = []
        bus_dict['edge_types'][bus_current] = []
    buses.append(bus_dict)

  return buses


def _compute_route_cost(graph, path):
  if path is None:
    return float('inf')

  total = 0.0
  for start, end in zip(path, path[1:]):
    idx = graph['connections'][start].index(end)
    total += graph['weights'][start][idx]
  return total


def _route_recommendation(route_cost, walking_cost):
  if route_cost == float('inf') and walking_cost == float('inf'):
    return 'no route could be found.'
  if walking_cost <= route_cost:
    return "you'd better go by foot."
  return "you'd better take the bus instead of walking."


if __name__ == '__main__':
  size = 10
  fixed_weight = 1

  map_graph = generate_square_city_graph(size, fixed_weight)
  buses_graph = generate_bus_line_square_city(size, fixed_weight)
  full_graph = merge_bus_and_map_graph(map_graph, buses_graph)

  # Prompt user for start and end nodes
  min_node, max_node = 0, size * size - 1

  def _prompt_node(prompt_text: str, default: int) -> int:
    while True:
      raw = input(f'{prompt_text} [{default}] (min {min_node}, max {max_node}): ').strip()
      if raw == '':
        return default
      try:
        val = int(raw)
        if min_node <= val <= max_node:
          return val
        else:
          print(f'Please enter a value between {min_node} and {max_node}.')
      except ValueError:
        print('Please enter a valid integer.')

  default_start, default_end = 3, 69
  start_node = _prompt_node('Enter start node', default_start)
  end_node = _prompt_node('Enter end node', default_end)

  route_solution = dijkstra(full_graph, start_node, end_node)
  route_solution_only_walking = dijkstra(map_graph, start_node, end_node)

  draw_graph(full_graph, route_solution, save_path='toy_city_graph_solution.html')
  draw_graph(map_graph, route_solution_only_walking, save_path='toy_city_graph_solution_walking.html')

  route_cost = _compute_route_cost(full_graph, route_solution)
  walking_cost = _compute_route_cost(map_graph, route_solution_only_walking)

  print('Route solution:', route_solution)
  print('Route cost:', route_cost)
  print('Route solution (walking):', route_solution_only_walking)
  print('Route cost (walking):', walking_cost)
  print(_route_recommendation(route_cost, walking_cost))
