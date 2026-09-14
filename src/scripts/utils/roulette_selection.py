import numpy as np


def roulette_wheel_selection(classes_probabilities):
  """Return a zero-based index sampled from the supplied class weights.

  ``classes_probabilities`` may be any one-dimensional numeric sequence with a
  positive total. Values are normalized before sampling, and the returned
  integer indexes that same sequence directly. NumPy's global random state is
  consumed, so callers may seed NumPy when reproducible experiments are needed.
  """
  classes_probabilities = np.array(classes_probabilities) / np.sum(classes_probabilities)
  return int(np.random.choice(len(classes_probabilities), p=classes_probabilities))
