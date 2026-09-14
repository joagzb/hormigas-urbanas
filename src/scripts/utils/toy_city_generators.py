"""Utility generators for simple square graphs used in tests.

These helpers are primarily intended for quick experimentation and unit tests.
They build a deterministic square grid and a vertical bus line in both directions to
validate routing algorithms.
"""

from ..utils.dijkstra import dijkstra
from ..utils.generators import generate_bus_line_square_city, generate_square_city_graph, merge_bus_and_map_graph
from ..utils.graph_visualizer import draw_graph


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
