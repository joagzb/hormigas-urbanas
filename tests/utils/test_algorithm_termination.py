import numpy as np
import pytest

from src.scripts.utils.algorithm_termination import global_best_patience_exhausted, update_global_best, validate_global_best


def test_strict_finite_improvement_resets_patience():
  improved, stagnant_epochs = update_global_best(5.0, 4.0, 3)

  assert improved is True
  assert stagnant_epochs == 0


@pytest.mark.parametrize('candidate', [5.0, 6.0, np.inf])
def test_equal_worse_and_infinite_candidates_increment_patience(candidate):
  improved, stagnant_epochs = update_global_best(5.0, candidate, 2)

  assert improved is False
  assert stagnant_epochs == 3


def test_patience_exhaustion_uses_configured_consecutive_count():
  assert not global_best_patience_exhausted(10, 9)
  assert global_best_patience_exhausted(10, 10)


@pytest.mark.parametrize('invalid_patience', [True, False, None, '10', 0, -1, 1.5, np.inf])
def test_global_best_patience_rejects_invalid_values(invalid_patience):
  with pytest.raises(ValueError, match='positive integer'):
    validate_global_best(invalid_patience)
