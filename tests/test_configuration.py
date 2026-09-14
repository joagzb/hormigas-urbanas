import json
import logging
from pathlib import Path

from src.configuration import algorithm_settings, graph_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_algorithm_settings_keys_and_values():
  expected_keys = {
    'ants',
    'f_ini',
    'f_min',
    'evaporation_rate',
    'epomax',
    'local_evaporation_rate',
    'transition_probability',
    'alfa',
    'beta',
    'global_best_patience',
    'bwas_restart_stagnation',
  }
  assert set(algorithm_settings.settings.keys()) == expected_keys

  rate_keys = {'f_min', 'evaporation_rate', 'local_evaporation_rate', 'transition_probability'}
  for key, value in algorithm_settings.settings.items():
    logger.info('algorithm_settings[%s] = %s', key, value)
    if key == 'f_ini':
      assert value is None
    elif key in rate_keys:
      assert 0 <= value <= 1
    elif key in {'ants', 'epomax', 'global_best_patience', 'bwas_restart_stagnation'}:
      assert value > 0
    else:
      assert value > 0

  assert algorithm_settings.settings['f_min'] == 1e-6
  assert algorithm_settings.settings['transition_probability'] == 0.9
  assert algorithm_settings.settings['alfa'] == 1.0
  assert algorithm_settings.settings['beta'] == 2.0
  assert algorithm_settings.settings['global_best_patience'] == 10
  assert algorithm_settings.settings['bwas_restart_stagnation'] == 8


def test_graph_settings_keys_and_values():
  expected_keys = {'wait_for_bus_cost', 'pay_for_bus_cost', 'bus_time_travel_cost', 'bus_get_off'}
  assert set(graph_settings.settings.keys()) == expected_keys

  rate_keys = {'bus_get_off'}
  for key, value in graph_settings.settings.items():
    logger.info('graph_settings[%s] = %s', key, value)
    if key in rate_keys:
      assert 0 < value <= 1
    else:
      assert value > 0


def test_notebook_uses_algorithm_settings_and_writes_html_without_inline_display():
  notebook = json.loads(Path('src/TPF.ipynb').read_text(encoding='utf-8'))
  source = ''.join(line for cell in notebook['cells'] if cell['cell_type'] == 'code' for line in cell['source'])

  expected_assignments = {
    "experiment = {'size': 10, 'start_node': 1, 'end_node': 97}",
    'prepare_routing_problem(full_graph',
    'ExperimentOutputWriter(full_problem.graph',
    "num_ants = settings['ants']",
    "evaporation_rate = settings['evaporation_rate']",
    "initial_pheromone_lvl = settings['f_ini']",
    "heuristic_weight = settings['alfa']",
    "pheromone_weight = settings['beta']",
    "epomax = settings['epomax']",
    "local_evap_rate = settings['local_evaporation_rate']",
    "transition_prob = settings['transition_probability']",
    "global_best_patience = settings['global_best_patience']",
    "bwas_restart_stagnation = settings['bwas_restart_stagnation']",
  }
  assert all(assignment in source for assignment in expected_assignments)
  assert algorithm_settings.settings['ants'] == 20
  assert algorithm_settings.settings['epomax'] == 100
  assert "experiment['ants']" not in source
  assert "experiment['epochs']" not in source
  assert source.count('num_ants') == 4
  assert source.count('epomax') == 5
  assert source.count('.write_animation(') == 3
  assert 'stride=' not in source
  assert 'max_frames=' not in source
  assert '.show(' not in source
  assert 'display(' not in source
  assert all(not cell.get('outputs') for cell in notebook['cells'])
  assert 'application/vnd.plotly' not in json.dumps(notebook)
  assert 'PheromoneHistoryWriter' not in source
  assert 'output_directory' not in source
  assert 'repository_root' not in source
  assert 'sys.path' not in source
  assert '.write_html(' not in source
  assert source.count('print(') == 9
  assert source.count('algorithm + history time:') == 3
  assert source.count("print('history load + HTML build/write time:'") == 3
  assert source.count("print('generated:'") == 3


def test_notebook_leaves_output_path_ownership_to_writer():
  repository_root = Path(__file__).resolve().parents[1]
  notebook = json.loads((repository_root / 'src' / 'TPF.ipynb').read_text(encoding='utf-8'))
  source = ''.join(line for cell in notebook['cells'] if cell['cell_type'] == 'code' for line in cell['source'])

  assert 'Path.cwd()' not in source
  assert 'resolve()' not in source
