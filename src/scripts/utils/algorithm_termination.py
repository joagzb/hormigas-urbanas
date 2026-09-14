import math
from collections import Counter
from numbers import Real


def validate_path_consensus_threshold(path_consensus_threshold):
  """Validate and return a path-consensus threshold in ``(0, 1]``."""
  if isinstance(path_consensus_threshold, bool) or not isinstance(path_consensus_threshold, Real) or not 0 < path_consensus_threshold <= 1:
    raise ValueError('path_consensus_threshold must be a number in (0, 1]')
  return path_consensus_threshold


def has_path_consensus(routes, costs, path_consensus_threshold):
  """Return whether enough finite ants completed the same exact route."""
  threshold = validate_path_consensus_threshold(path_consensus_threshold)
  finite_routes = [tuple(route) for route, cost in zip(routes, costs) if route is not None and math.isfinite(cost)]
  if not finite_routes:
    return False
  consensus_count = Counter(finite_routes).most_common(1)[0][1]
  required_count = math.ceil(threshold * len(finite_routes))
  return consensus_count >= required_count


def has_stable_iteration_best_cost(previous_cost, current_cost):
  """Return whether consecutive finite iteration-best costs are exactly equal."""
  return previous_cost is not None and math.isfinite(previous_cost) and math.isfinite(current_cost) and previous_cost == current_cost
