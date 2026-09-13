settings = {
  'ants': 50,
  'f_ini': None,  # derive each algorithm's documented tau0
  'f_min': 1e-6,  # BWAS floor below the derived pheromone level for this city
  'evaporation_rate': 0.1,  # (p) pheromone evaporation level - lowered to improve memory retention
  'epomax': 500,
  'local_evaporation_rate': 0.1,  # rho - lowered to be consistent
  'transition_probability': 0.9,  # q0 - probability of ACS exploitation
  'alfa': 1.0,  # heuristic_weight - exponent on pheromone
  'beta': 2.0,  # pheromone_weight - exponent on inverse selection cost
  'acs_stagnation_epochs': 25,
  'bwas_stagnation_epochs': 50,
  'bwas_restart_stagnation': 8,
}

presets = {
  # Profile A — bus‑friendly balanced
  'bus_friendly': {'ants': 80, 'evaporation_rate': 0.35, 'f_ini': 0.5, 'alfa': 1.0, 'beta': 0.35, 'epomax': 2000, 'local_evaporation_rate': 0.12, 'transition_probability': 0.15},
  # Profile B — stronger exploration early
  'explore_strong': {
    'ants': 120,
    'evaporation_rate': 0.25,
    'f_ini': 0.6,
    'alfa': 1.2,
    'beta': 0.25,
    'epomax': 3000,
    'local_evaporation_rate': 0.10,
    'transition_probability': 0.10,
  },
}


def load_profile(name: str) -> dict:
  """Return a copy of the base settings updated with a named preset.

  The returned dict includes keys used across ACO/ACS implementations:
  - ants, evaporation_rate, f_ini, alfa, beta, epomax,
    local_evaporation_rate, transition_probability, and stagnation limits
  """
  base = dict(settings)
  profile = presets.get(name)
  if profile:
    base.update(profile)
  return base
