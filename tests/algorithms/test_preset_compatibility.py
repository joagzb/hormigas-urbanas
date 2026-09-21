import numpy as np
import pytest

from src.configuration.algorithm_settings import load_profile
from src.scripts.ant_best_worst.ant_colony_best_worst import ABW
from src.scripts.ant_colony_simple_ACO.ant_colony_optimization import ACO
from src.scripts.ant_colony_system.ant_colony_system import ACS
from src.scripts.utils.generators import generate_bus_line_square_city, generate_square_city_graph, merge_bus_and_map_graph


@pytest.fixture(scope='module')
def multimodal_city_20():
  return merge_bus_and_map_graph(generate_square_city_graph(20, 1), generate_bus_line_square_city(20, 1))


def test_aco_bus_friendly_preset_finds_finite_endpoint_route(multimodal_city_20):
  profile = load_profile('bus_friendly')
  np.random.seed(0)

  path, cost, _, _ = ACO(
    multimodal_city_20,
    1,
    397,
    profile['ants'],
    profile['evaporation_rate'],
    profile['f_ini'],
    profile['alfa'],
    profile['beta'],
    profile['epomax'],
    global_best_patience=profile['global_best_patience'],
  )

  assert path is not None
  assert path[0] == 1
  assert path[-1] == 397
  assert np.isfinite(cost)


def test_acs_multimodal_preset_finds_finite_endpoint_route(multimodal_city_20):
  profile = load_profile('acs_multimodal')
  np.random.seed(0)

  path, cost, _, _ = ACS(
    multimodal_city_20,
    1,
    397,
    profile['ants'],
    profile['evaporation_rate'],
    profile['local_evaporation_rate'],
    profile['transition_probability'],
    profile['f_ini'],
    profile['alfa'],
    profile['beta'],
    profile['epomax'],
    global_best_patience=profile['global_best_patience'],
  )

  assert path is not None
  assert path[0] == 1
  assert path[-1] == 397
  assert np.isfinite(cost)


def test_bwas_aggressive_preset_finds_finite_endpoint_route(multimodal_city_20):
  profile = load_profile('bwas_aggressive')
  np.random.seed(0)

  path, cost, _, _ = ABW(
    multimodal_city_20,
    1,
    397,
    profile['ants'],
    profile['evaporation_rate'],
    profile['epomax'],
    profile['f_ini'],
    profile['alfa'],
    profile['beta'],
    worst_penalty_rate=profile['worst_penalty_rate'],
    mutation_probability=profile['mutation_probability'],
    mutation_scale=profile['mutation_scale'],
    restart_stagnation=profile['bwas_restart_stagnation'],
    min_pheromone_lvl=profile['f_min'],
    global_best_patience=profile['global_best_patience'],
  )

  assert path is not None
  assert path[0] == 1
  assert path[-1] == 397
  assert np.isfinite(cost)
