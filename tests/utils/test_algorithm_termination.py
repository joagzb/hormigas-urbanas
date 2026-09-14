import numpy as np
import pytest

from src.scripts.utils.algorithm_termination import has_path_consensus, has_stable_iteration_best_cost


def test_seventeen_of_twenty_finite_ants_reach_eighty_five_percent_consensus():
  routes = [[0, 1]] * 17 + [[0, 2], [0, 3], [0, 4]]

  assert has_path_consensus(routes, [1.0] * 20, 0.85)


def test_lost_ants_are_excluded_from_consensus_denominator():
  routes = [[0, 1]] * 17 + [[0, 2], None, None]
  costs = [1.0] * 18 + [np.inf, np.inf]

  assert has_path_consensus(routes, costs, 0.85)


def test_equal_cost_different_paths_do_not_form_path_consensus():
  routes = [[0, 1]] * 8 + [[0, 2]] * 2

  assert not has_path_consensus(routes, [1.0] * 10, 0.85)


@pytest.mark.parametrize('invalid_threshold', [True, False, None, '0.85', 0, -0.1, 1.01, np.inf, np.nan])
def test_path_consensus_rejects_invalid_threshold(invalid_threshold):
  with pytest.raises(ValueError, match=r'\(0, 1\]'):
    has_path_consensus([[0, 1]], [1.0], invalid_threshold)


def test_stable_iteration_best_requires_exact_consecutive_finite_costs():
  assert has_stable_iteration_best_cost(2.0, 2.0)
  assert not has_stable_iteration_best_cost(None, 2.0)
  assert not has_stable_iteration_best_cost(np.inf, np.inf)
  assert not has_stable_iteration_best_cost(2.0, 2.0 + 1e-12)
