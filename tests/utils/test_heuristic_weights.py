import numpy as np
import pytest

from src.scripts.utils.heuristic_weights import normalize_weights_for_selection


def test_normalization_uses_edge_semantics_not_matching_numeric_costs():
  weights = np.array([1.4, 1.4, 0.01, 0.01])
  edge_types = np.array(['walk', 'board', 'ride', 'alight'], dtype=object)

  adjusted = normalize_weights_for_selection(weights, edge_types)

  assert adjusted[0] == pytest.approx(1.4)
  assert adjusted[1] < adjusted[0]
  assert adjusted[2] == pytest.approx(0.01)
  assert adjusted[3] > adjusted[2]


def test_zero_cost_alighting_gets_positive_selection_only_weight():
  weights = np.array([0.0])

  adjusted = normalize_weights_for_selection(weights, np.array(['alight'], dtype=object))

  assert adjusted[0] > 0
  assert weights[0] == 0.0
