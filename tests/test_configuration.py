import json
import logging
from pathlib import Path

import pytest

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
    'aco_global_best_patience',
    'path_consensus_threshold',
    'bwas_restart_stagnation',
  }
  assert set(algorithm_settings.settings.keys()) == expected_keys

  rate_keys = {'f_min', 'evaporation_rate', 'local_evaporation_rate', 'transition_probability', 'path_consensus_threshold'}
  for key, value in algorithm_settings.settings.items():
    logger.info('algorithm_settings[%s] = %s', key, value)
    if key == 'f_ini':
      assert value is None
    elif key in rate_keys:
      assert 0 <= value <= 1
    elif key in {'ants', 'epomax', 'aco_global_best_patience', 'bwas_restart_stagnation'}:
      assert value > 0
    else:
      assert value > 0

  assert algorithm_settings.settings['f_min'] == 1e-6
  assert algorithm_settings.settings['transition_probability'] == 0.9
  assert algorithm_settings.settings['alfa'] == 1.0
  assert algorithm_settings.settings['beta'] == 2.0
  assert algorithm_settings.settings['aco_global_best_patience'] == 10
  assert algorithm_settings.settings['path_consensus_threshold'] == 0.85
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
    'working_directory = Path.cwd().resolve()',
    "if (candidate / 'src' / 'TPF.ipynb').is_file()",
    "output_directory = (repository_root / 'tmp').resolve()",
    'output_directory.mkdir(parents=True, exist_ok=True)',
    "num_ants = settings['ants']",
    "evaporation_rate = settings['evaporation_rate']",
    "initial_pheromone_lvl = settings['f_ini']",
    "heuristic_weight = settings['alfa']",
    "pheromone_weight = settings['beta']",
    "epomax = settings['epomax']",
    "local_evap_rate = settings['local_evaporation_rate']",
    "transition_prob = settings['transition_probability']",
    "aco_global_best_patience = settings['aco_global_best_patience']",
    "path_consensus_threshold = settings['path_consensus_threshold']",
    "bwas_restart_stagnation = settings['bwas_restart_stagnation']",
  }
  assert all(assignment in source for assignment in expected_assignments)
  assert algorithm_settings.settings['ants'] == 20
  assert algorithm_settings.settings['epomax'] == 100
  assert "experiment['ants']" not in source
  assert "experiment['epochs']" not in source
  assert source.count('num_ants') == 4
  assert source.count('epomax') == 5
  assert source.count('draw_pheromone_history(') == 3
  assert source.count('show=False') == 3
  assert '.show(' not in source
  assert 'display(' not in source
  assert all(not cell.get('outputs') for cell in notebook['cells'])
  assert 'application/vnd.plotly' not in json.dumps(notebook)
  assert "aco_history_path = output_directory / 'aco_pheromone_history.jsonl'" in source
  assert "acs_history_path = output_directory / 'acs_pheromone_history.jsonl'" in source
  assert "bwas_history_path = output_directory / 'bwas_pheromone_history.jsonl'" in source
  assert "aco_html_path = output_directory / 'aco_pheromone_animation.html'" in source
  assert "acs_html_path = output_directory / 'acs_pheromone_animation.html'" in source
  assert "bwas_html_path = output_directory / 'bwas_pheromone_animation.html'" in source
  assert source.count('.write_html(') == 3
  assert source.count('print(') == 9
  assert source.count('algorithm + history time:') == 3
  assert source.count("print('history load + HTML build/write time:'") == 3
  assert source.count("print('generated:'") == 3


@pytest.mark.parametrize('working_directory_name', ['', 'src'])
def test_notebook_resolves_all_outputs_under_repository_tmp(monkeypatch, working_directory_name):
  repository_root = Path(__file__).resolve().parents[1]
  notebook = json.loads((repository_root / 'src' / 'TPF.ipynb').read_text(encoding='utf-8'))
  bootstrap_source = ''.join(notebook['cells'][0]['source']).split('%pip', 1)[0]
  namespace = {}
  monkeypatch.chdir(repository_root / working_directory_name)

  exec(compile(bootstrap_source, 'TPF-bootstrap', 'exec'), namespace)

  output_directory = namespace['output_directory']
  assert output_directory == (repository_root / 'tmp').resolve()
  for filename in (
    'aco_pheromone_history.jsonl',
    'acs_pheromone_history.jsonl',
    'bwas_pheromone_history.jsonl',
    'aco_pheromone_animation.html',
    'acs_pheromone_animation.html',
    'bwas_pheromone_animation.html',
  ):
    assert (output_directory / filename).is_relative_to(repository_root / 'tmp')
