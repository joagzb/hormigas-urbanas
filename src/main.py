"""Public routing boundary and quick interactive algorithm comparison."""

import copy
from dataclasses import dataclass
from pathlib import Path

from src.scripts.utils.graph_visualizer import ExperimentOutputWriter

if __package__:
  from .configuration.algorithm_settings import load_profile, presets, settings
  from .scripts.ant_best_worst.ant_colony_best_worst import ABW
  from .scripts.ant_colony_simple_ACO.ant_colony_optimization import ACO
  from .scripts.ant_colony_system.ant_colony_system import ACS
  from .scripts.utils.dijkstra import dijkstra
  from .scripts.utils.generators import generate_bus_line_square_city, generate_square_city_graph, merge_bus_and_map_graph, validate_graph
  from .scripts.utils.graph_visualizer import draw_graph
else:
  from configuration.algorithm_settings import load_profile, presets, settings
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


def _prompt_preset(prompt_text):
  """Prompt until the user supplies an algorithm preset."""
  while True:
    print('Available Presets:')
    for key in presets.keys():
      print(f'  - {key}')

    value = input(f'\n{prompt_text} [Press Enter for default]: ').strip()

    if value == '':
      return dict(settings)

    profile = presets.get(value)
    while not profile:
      print('Please enter a valid preset or empty for default.')

    return load_profile(value)


def _print_algorithm_result(name, result):
  path, cost, _, epochs = result
  print(f'{name} route solution:', path)
  print(f'{name} cost:', cost)
  print(f'{name} epochs:', epochs)


def _print_preset(algorithm_settings):
  for key in sorted(algorithm_settings):
    print(f'  {key}: {algorithm_settings[key]}')


def _write_route_html(graph, reference_route, algorithm_name, result, ant_count):
  output_directory = Path(__file__).resolve().parents[1] / 'tmp'
  output_directory.mkdir(parents=True, exist_ok=True)
  display_name = 'BWAS' if algorithm_name == 'ABW' else algorithm_name
  metadata = f'{display_name} route ({ant_count} ants)'
  draw_graph(graph, result[0], save_path=output_directory / f'{display_name.lower()}_route.html', reference_path=reference_route, title=metadata, route_label=metadata)


def main(argv=None):
  size = 20
  fixed_weight = 1

  algorithm_settings = _prompt_preset('preset option')
  _print_preset(algorithm_settings)

  map_graph = generate_square_city_graph(size, fixed_weight)
  buses_graph = generate_bus_line_square_city(size, fixed_weight)
  full_graph = merge_bus_and_map_graph(map_graph, buses_graph)

  start_node = _prompt_node('Enter start node', 3, 0, size * size - 1)
  end_node = _prompt_node('Enter end node', 69, 0, size * size - 1)

  problem = prepare_routing_problem(full_graph, start_node, end_node)

  reference_route = dijkstra(problem.graph, start_node, end_node)

  while start_node == end_node:
    end_node = _prompt_node('Enter end node different from start node', 69, 0, size * size - 1)

  aco_output = ExperimentOutputWriter(full_graph, 'aco')
  acs_output = ExperimentOutputWriter(full_graph, 'acs')
  bwas_output = ExperimentOutputWriter(full_graph, 'bwas')

  aco_result = ACO(
    problem.graph,
    start_node,
    end_node,
    algorithm_settings['ants'],
    algorithm_settings['evaporation_rate'],
    algorithm_settings['f_ini'],
    algorithm_settings['alfa'],
    algorithm_settings['beta'],
    algorithm_settings['epomax'],
    global_best_patience=algorithm_settings['global_best_patience'],
    epoch_callback=aco_output,
  )
  acs_result = ACS(
    problem.graph,
    start_node,
    end_node,
    algorithm_settings['ants'],
    algorithm_settings['evaporation_rate'],
    algorithm_settings['local_evaporation_rate'],
    algorithm_settings['transition_probability'],
    algorithm_settings['f_ini'],
    algorithm_settings['alfa'],
    algorithm_settings['beta'],
    algorithm_settings['epomax'],
    global_best_patience=algorithm_settings['global_best_patience'],
    epoch_callback=acs_output,
  )
  bwas_result = ABW(
    problem.graph,
    start_node,
    end_node,
    algorithm_settings['ants'],
    algorithm_settings['evaporation_rate'],
    algorithm_settings['epomax'],
    algorithm_settings['f_ini'],
    algorithm_settings['alfa'],
    algorithm_settings['beta'],
    worst_penalty_rate=algorithm_settings.get('worst_penalty_rate'),
    mutation_probability=algorithm_settings.get('mutation_probability', 0.05),
    mutation_scale=algorithm_settings.get('mutation_scale', 2.0),
    restart_stagnation=algorithm_settings.get('bwas_restart_stagnation'),
    min_pheromone_lvl=algorithm_settings.get('f_min'),
    global_best_patience=algorithm_settings['global_best_patience'],
    epoch_callback=bwas_output,
  )

  for name, result in (('ACO', aco_result), ('ACS', acs_result), ('ABW', bwas_result)):
    _print_algorithm_result(name, result)

  aco_output.write_animation(reference_path=reference_route, title='ACO pheromone evolution', route_label=f'ACO global best-found route ({algorithm_settings["ants"]} ants)')
  acs_output.write_animation(reference_path=reference_route, title='ACS pheromone evolution', route_label=f'ACS global best-found route ({algorithm_settings["ants"]} ants)')
  bwas_output.write_animation(reference_path=reference_route, title='BWAS pheromone evolution', route_label=f'BWAS global best-found route ({algorithm_settings["ants"]} ants)')


if __name__ == '__main__':
  main()