def notify_stage(callback, *, epoch, stage, pheromones, iteration_best_path, iteration_best_cost, global_best_path, global_best_cost, restarted=False):
  """Send one independent post-update observation when a callback exists."""
  if callback is None:
    return

  callback(
    {
      'epoch': epoch,
      'stage': stage,
      'pheromones': {node: values.copy() for node, values in pheromones.items()},
      'iteration_best_path': _copy_path(iteration_best_path),
      'iteration_best_cost': iteration_best_cost,
      'global_best_path': _copy_path(global_best_path),
      'global_best_cost': global_best_cost,
      'restarted': restarted,
    }
  )


def _copy_path(path):
  return None if path is None else path.copy()
