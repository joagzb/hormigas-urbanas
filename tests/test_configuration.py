import ast
import json
import logging
import math
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


def test_algorithm_profiles_have_valid_domains_and_compatible_restart_patience():
  for name in algorithm_settings.presets:
    profile = algorithm_settings.load_profile(name)

    assert isinstance(profile['ants'], int) and not isinstance(profile['ants'], bool) and profile['ants'] > 0
    assert isinstance(profile['epomax'], int) and not isinstance(profile['epomax'], bool) and profile['epomax'] > 0
    assert isinstance(profile['global_best_patience'], int) and profile['global_best_patience'] > 0
    assert profile['f_ini'] is None or math.isfinite(profile['f_ini']) and profile['f_ini'] > 0
    assert math.isfinite(profile['f_min']) and profile['f_min'] > 0
    if profile['f_ini'] is not None:
      assert profile['f_min'] <= profile['f_ini']
    for rate_key in ('evaporation_rate', 'local_evaporation_rate', 'transition_probability'):
      assert 0 <= profile[rate_key] <= 1
    for exponent_key in ('alfa', 'beta'):
      assert math.isfinite(profile[exponent_key]) and profile[exponent_key] >= 0
    if profile['bwas_restart_stagnation'] > 0:
      assert profile['bwas_restart_stagnation'] < profile['global_best_patience']


def test_algorithm_specific_profiles_expose_compatible_values():
  acs = algorithm_settings.load_profile('acs_multimodal')
  bwas = algorithm_settings.load_profile('bwas_aggressive')

  assert {key: acs[key] for key in ('ants', 'evaporation_rate', 'f_ini', 'local_evaporation_rate', 'transition_probability', 'alfa', 'beta', 'epomax', 'global_best_patience')} == {
    'ants': 80,
    'evaporation_rate': 0.20,
    'f_ini': None,
    'local_evaporation_rate': 0.10,
    'transition_probability': 0.15,
    'alfa': 1.0,
    'beta': 0.5,
    'epomax': 1500,
    'global_best_patience': 30,
  }
  assert bwas['global_best_patience'] == 30
  assert 0 <= bwas['worst_penalty_rate'] <= 1
  assert 0 <= bwas['mutation_probability'] <= 1
  assert math.isfinite(bwas['mutation_scale']) and bwas['mutation_scale'] >= 0


def test_load_profile_returns_fresh_copies_and_preserves_unknown_profile_behavior():
  loaded = algorithm_settings.load_profile('acs_multimodal')
  loaded['ants'] = -1

  assert algorithm_settings.load_profile('acs_multimodal')['ants'] == 80
  assert algorithm_settings.presets['acs_multimodal']['ants'] == 80
  assert algorithm_settings.load_profile('unknown-profile') == algorithm_settings.settings
  assert algorithm_settings.load_profile('unknown-profile') is not algorithm_settings.settings


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
  source = '\n'.join(''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code')
  tree = ast.parse(source)

  loaded_profiles = {}
  for node in ast.walk(tree):
    if (
      isinstance(node, ast.Assign)
      and len(node.targets) == 1
      and isinstance(node.targets[0], ast.Name)
      and isinstance(node.value, ast.Call)
      and isinstance(node.value.func, ast.Name)
      and node.value.func.id == 'load_profile'
    ):
      loaded_profiles[node.targets[0].id] = node.value.args[0].value
  assert loaded_profiles == {'aco_settings': 'bus_friendly', 'acs_settings': 'acs_multimodal', 'bwas_settings': 'bwas_aggressive'}

  algorithm_calls = {node.func.id: node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'ACO', 'ACS', 'ABW'}}
  assert [ast.unparse(argument) for argument in algorithm_calls['ACO'].args[3:]] == [
    "aco_settings['ants']",
    "aco_settings['evaporation_rate']",
    "aco_settings['f_ini']",
    "aco_settings['alfa']",
    "aco_settings['beta']",
    "aco_settings['epomax']",
  ]
  assert [ast.unparse(argument) for argument in algorithm_calls['ACS'].args[3:]] == [
    "acs_settings['ants']",
    "acs_settings['evaporation_rate']",
    "acs_settings['local_evaporation_rate']",
    "acs_settings['transition_probability']",
    "acs_settings['f_ini']",
    "acs_settings['alfa']",
    "acs_settings['beta']",
    "acs_settings['epomax']",
  ]
  assert [ast.unparse(argument) for argument in algorithm_calls['ABW'].args[3:]] == [
    "bwas_settings['ants']",
    "bwas_settings['evaporation_rate']",
    "bwas_settings['epomax']",
    "bwas_settings['f_ini']",
    "bwas_settings['alfa']",
    "bwas_settings['beta']",
  ]
  assert {keyword.arg: ast.unparse(keyword.value) for keyword in algorithm_calls['ACO'].keywords} == {
    'global_best_patience': "aco_settings['global_best_patience']",
    'epoch_callback': 'aco_output',
  }
  assert {keyword.arg: ast.unparse(keyword.value) for keyword in algorithm_calls['ACS'].keywords} == {
    'global_best_patience': "acs_settings['global_best_patience']",
    'epoch_callback': 'acs_output',
  }
  bwas_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in algorithm_calls['ABW'].keywords}
  assert bwas_keywords == {
    'worst_penalty_rate': "bwas_settings['worst_penalty_rate']",
    'mutation_probability': "bwas_settings['mutation_probability']",
    'mutation_scale': "bwas_settings['mutation_scale']",
    'restart_stagnation': "bwas_settings['bwas_restart_stagnation']",
    'min_pheromone_lvl': "bwas_settings['f_min']",
    'global_best_patience': "bwas_settings['global_best_patience']",
    'epoch_callback': 'bwas_output',
  }
  assert algorithm_settings.settings['ants'] == 20
  assert algorithm_settings.settings['epomax'] == 100
  assert "experiment['ants']" not in source
  assert "experiment['epochs']" not in source
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
  assert all(label in source for label in ("print('ACO result:'", "print('ACS result:'", "print('BWAS result:'"))


def test_notebook_leaves_output_path_ownership_to_writer():
  repository_root = Path(__file__).resolve().parents[1]
  notebook = json.loads((repository_root / 'src' / 'TPF.ipynb').read_text(encoding='utf-8'))
  source = ''.join(line for cell in notebook['cells'] if cell['cell_type'] == 'code' for line in cell['source'])

  assert 'Path.cwd()' not in source
  assert 'resolve()' not in source
