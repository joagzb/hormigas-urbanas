import math
from numbers import Integral


def validate_global_best(global_best_patience):
  """Return a positive integer global-best stagnation limit.

  The value counts consecutive completed epochs without a strict improvement
  to a finite global-best cost. Booleans are rejected even though they are
  integer subclasses in Python.
  """
  if isinstance(global_best_patience, bool) or not isinstance(global_best_patience, Integral) or global_best_patience <= 0:
    raise ValueError('global_best_patience must be a positive integer')
  return global_best_patience


def update_global_best(global_best_cost, candidate_cost, epochs_without_improvement):
  """Update strict global-best improvement and consecutive stagnation state.

  Inputs are the retained best cost, the current epoch's best cost, and the
  previous consecutive non-improvement count. The returned tuple contains an
  improvement flag and the next count. Infinite candidates never improve.
  """
  improved = math.isfinite(candidate_cost) and candidate_cost < global_best_cost
  if improved:
    next_epochs_without_improvement = 0
  else:
    next_epochs_without_improvement = epochs_without_improvement + 1
  return improved, next_epochs_without_improvement


def global_best_patience_exhausted(global_best_patience, epochs_without_improvement):
  """Return whether the validated strict-improvement patience is exhausted."""
  return epochs_without_improvement >= validate_global_best(global_best_patience)
