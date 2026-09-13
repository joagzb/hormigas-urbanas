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
    'acs_stagnation_epochs',
    'bwas_stagnation_epochs',
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
    elif key in {'ants', 'epomax', 'acs_stagnation_epochs', 'bwas_stagnation_epochs', 'bwas_restart_stagnation'}:
      assert value > 0
    else:
      assert value > 0

  assert algorithm_settings.settings['f_min'] == 1e-6
  assert algorithm_settings.settings['transition_probability'] == 0.9
  assert algorithm_settings.settings['alfa'] == 1.0
  assert algorithm_settings.settings['beta'] == 2.0
  assert algorithm_settings.settings['acs_stagnation_epochs'] == 25
  assert algorithm_settings.settings['bwas_stagnation_epochs'] == 50
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


def test_notebook_uses_explicit_demo_and_full_experiment_profiles():
  notebook = json.loads(Path('src/TPF.ipynb').read_text(encoding='utf-8'))
  source = ''.join(line for cell in notebook['cells'] if cell['cell_type'] == 'code' for line in cell['source'])

  expected_assignments = {
    'RUN_FULL_EXPERIMENT = False',
    "FAST_DEMO = {'size': 10, 'ants': 10, 'epochs': 10",
    "FULL_EXPERIMENT = {'size': 20, 'ants': 50, 'epochs': 500",
    'experiment = FULL_EXPERIMENT if RUN_FULL_EXPERIMENT else FAST_DEMO',
    "num_ants = experiment['ants']",
    "evaporation_rate = settings['evaporation_rate']",
    "initial_pheromone_lvl = settings['f_ini']",
    "heuristic_weight = settings['alfa']",
    "pheromone_weight = settings['beta']",
    "epomax = experiment['epochs']",
    "local_evap_rate = settings['local_evaporation_rate']",
    "transition_prob = settings['transition_probability']",
    "acs_stagnation_epochs = settings['acs_stagnation_epochs']",
    "bwas_stagnation_epochs = settings['bwas_stagnation_epochs']",
    "bwas_restart_stagnation = settings['bwas_restart_stagnation']",
  }
  assert all(assignment in source for assignment in expected_assignments)
