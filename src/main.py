"""Public routing boundary and quick interactive algorithm comparison."""

import copy
from dataclasses import dataclass

if __package__:
  from .configuration.algorithm_settings import settings
  from .scripts.ant_best_worst.ant_colony_best_worst import ABW
  from .scripts.ant_colony_simple_ACO.ant_colony_optimization import ACO
  from .scripts.ant_colony_system.ant_colony_system import ACS
  from .scripts.utils.dijkstra import dijkstra
  from .scripts.utils.generators import generate_bus_line_square_city, generate_square_city_graph, merge_bus_and_map_graph, validate_graph
  from .scripts.utils.graph_visualizer import draw_graph
else:
  from configuration.algorithm_settings import settings
  from scripts.ant_best_worst.ant_colony_best_worst import ABW
  from scripts.ant_colony_simple_ACO.ant_colony_optimization import ACO
  from scripts.ant_colony_system.ant_colony_system import ACS
  from scripts.utils.dijkstra import dijkstra
  from scripts.utils.generators import generate_bus_line_square_city, generate_square_city_graph, merge_bus_and_map_graph, validate_graph
  from scripts.utils.graph_visualizer import draw_graph


def _requires_edge_types(graph):
  return bool(graph.get('buses')) or any(isinstance(node, str) and node.startswith('bus:') for node in graph.get('node_index', []))


@dataclass(frozen=True)
class RoutingProblem:
  """A validated graph and endpoint pair ready for algorithm execution.

  ``graph`` contains opaque node IDs and aligned connection, weight, and edge
  type rows. ``run`` returns a zero-epoch route for identical endpoints or
  delegates to the supplied colony function without changing its result.
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

  Walking-only graphs without ``edge_types`` receive aligned ``walk`` rows.
  Multimodal graphs must provide explicit edge types. The returned problem owns
  a deep copy of the supplied graph so neither side can mutate the other.
  """
  prepared_graph = copy.deepcopy(graph)
  if 'edge_types' not in graph:
    if _requires_edge_types(graph):
      raise ValueError('Multimodal graphs require explicit edge_types')
    prepared_graph['edge_types'] = {node: ['walk'] * len(neighbors) for node, neighbors in graph.get('connections', {}).items()}

  validate_graph(prepared_graph)
  nodes = prepared_graph['node_index']
  if start_node not in nodes or end_node not in nodes:
    raise ValueError('start_node and end_node must exist in the graph')
  return RoutingProblem(prepared_graph, start_node, end_node)


def _prompt_node(prompt_text, default, min_node, max_node):
  """Prompt until the user supplies an integer within the map-node range."""
  while True:
    raw = input(f'{prompt_text} [{default}] (min {min_node}, max {max_node}): ').strip()
    if raw == '':
      return default
    try:
      value = int(raw)
    except ValueError:
      print('Please enter a valid integer.')
      continue
    if min_node <= value <= max_node:
      return value
    print(f'Please enter a value between {min_node} and {max_node}.')


def _compute_route_cost(graph, path):
  """Return the sum of aligned edge weights, or infinity for no route."""
  if path is None:
    return float('inf')

  total = 0.0
  for start, end in zip(path, path[1:]):
    edge_index = graph['connections'][start].index(end)
    total += graph['weights'][start][edge_index]
  return total


def _route_recommendation(route_cost, walking_cost):
  """Recommend the lower-cost Dijkstra option, preferring walking on ties."""
  if route_cost == float('inf') and walking_cost == float('inf'):
    return 'no route could be found.'
  if walking_cost <= route_cost:
    return "you'd better go by foot."
  return "you'd better take the bus instead of walking."


def _print_algorithm_result(name, result):
  path, cost, elapsed, epochs = result
  print(f'{name} route:', path)
  print(f'{name} cost:', cost)
  print(f'{name} time:', elapsed)
  print(f'{name} epochs:', epochs)


def main():
  """Run the lightweight interactive Dijkstra, ACO, ACS, and ABW demo."""
  size = 20
  fixed_weight = 1
  map_graph = generate_square_city_graph(size, fixed_weight)
  buses_graph = generate_bus_line_square_city(size, fixed_weight)
  full_graph = merge_bus_and_map_graph(map_graph, buses_graph)

  start_node = _prompt_node('Enter start node', 3, 0, size * size - 1)
  end_node = _prompt_node('Enter end node', 69, 0, size * size - 1)
  
  problem = prepare_routing_problem(full_graph, start_node, end_node)

  route = dijkstra(problem.graph, start_node, end_node)
  walking_route = dijkstra(map_graph, start_node, end_node)
  route_cost = _compute_route_cost(problem.graph, route)
  walking_cost = _compute_route_cost(map_graph, walking_route)

  draw_graph(problem.graph, route, save_path='toy_city_graph_solution.html')
  draw_graph(map_graph, walking_route, save_path='toy_city_graph_solution_walking.html')

  print('Dijkstra route:', route)
  print('Dijkstra cost:', route_cost)
  print('Dijkstra walking route:', walking_route)
  print('Dijkstra walking cost:', walking_cost)
  print(_route_recommendation(route_cost, walking_cost))

  _print_algorithm_result(
    'ACO',
    problem.run(
      ACO,
      settings['ants'],
      settings['evaporation_rate'],
      settings['f_ini'],
      settings['alfa'],
      settings['beta'],
      settings['epomax'],
      global_best_patience=settings['global_best_patience'],
    ),
  )
  _print_algorithm_result(
    'ACS',
    problem.run(
      ACS,
      settings['ants'],
      settings['evaporation_rate'],
      settings['local_evaporation_rate'],
      settings['transition_probability'],
      settings['f_ini'],
      settings['alfa'],
      settings['beta'],
      settings['epomax'],
      global_best_patience=settings['global_best_patience'],
    ),
  )
  _print_algorithm_result(
    'ABW',
    problem.run(
      ABW,
      settings['ants'],
      settings['evaporation_rate'],
      settings['epomax'],
      settings['f_ini'],
      settings['alfa'],
      settings['beta'],
      worst_penalty_rate=settings['worst_penalty_rate'],
      mutation_probability=settings['mutation_probability'],
      mutation_scale=settings['mutation_scale'],
      restart_stagnation=settings['bwas_restart_stagnation'],
      min_pheromone_lvl=settings['f_min'],
      global_best_patience=settings['global_best_patience'],
    ),
  )


if __name__ == '__main__':
  main()
