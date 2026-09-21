import math
from numbers import Integral


def validate_global_best(global_best_patience):
  """validate a positive integer global-best stagnation limit."""
  if isinstance(global_best_patience, bool) or not isinstance(global_best_patience, Integral) or global_best_patience <= 0:
    raise ValueError('global_best_patience must be a positive integer')
  return global_best_patience


def update_global_best(global_best_cost, candidate_cost, epochs_without_improvement):
  """Update strict global-best improvement and consecutive stagnation state.

  Check if the global best cost solution has improved or not to tell
  the algorithm to stop. The returned tuple contains an
  improvement flag and the next count. Infinite candidates never improve.
  """
  is_global_solution_improved = math.isfinite(candidate_cost) and candidate_cost < global_best_cost
  if is_global_solution_improved:
    next_epochs_without_improvement = 0
  else:
    next_epochs_without_improvement = epochs_without_improvement + 1

  return is_global_solution_improved, next_epochs_without_improvement


def global_best_patience_exhausted(global_best_patience, epochs_without_improvement):
  """Return whether the validated strict-improvement patience is exhausted."""
  return epochs_without_improvement >= validate_global_best(global_best_patience)
