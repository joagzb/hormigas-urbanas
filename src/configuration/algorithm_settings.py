settings = {
  'ants': 20,
  'f_ini': None,  # derive each algorithm's documented tau0
  'f_min': 1e-6,  # BWAS floor below the derived pheromone level for this city
  'evaporation_rate': 0.1,  # (p) pheromone evaporation level - lowered to improve memory retention
  'epomax': 100,
  'local_evaporation_rate': 0.1,  # rho - lowered to be consistent
  'transition_probability': 0.9,  # q0 - probability of ACS exploitation
  'alfa': 1.0,  # heuristic_weight - exponent on pheromone
  'beta': 2.0,  # pheromone_weight - exponent on inverse selection cost
  'global_best_patience': 10,
  'bwas_restart_stagnation': 8,
}

presets = {
  # Profile A — Bus-friendly balanced
  'bus_friendly': {
    'ants': 80,
    'evaporation_rate': 0.35,
    'f_ini': 0.5,
    'f_min': 1e-6,
    'alfa': 1.0,
    'beta': 0.35,
    'epomax': 2000,
    'local_evaporation_rate': 0.12,
    'transition_probability': 0.15,
    'global_best_patience': 15,
    'bwas_restart_stagnation': 8,
  },
  # Profile B — Stronger exploration early
  'explore_strong': {
    'ants': 80,
    'evaporation_rate': 0.25,
    'f_ini': 0.6,
    'f_min': 1e-6,
    'alfa': 1.2,
    'beta': 0.25,
    'epomax': 3000,
    'local_evaporation_rate': 0.10,
    'transition_probability': 0.10,
    'global_best_patience': 20,
    'bwas_restart_stagnation': 8,
  },
  # Profile C — Standard ACS (Dorigo & Gambardella benchmark)
  'classic_acs': {
    'ants': 10,
    'evaporation_rate': 0.10,
    'f_ini': None,
    'f_min': 1e-6,
    'alfa': 1.0,
    'beta': 2.0,
    'epomax': 1000,
    'local_evaporation_rate': 0.10,
    'transition_probability': 0.90,
    'global_best_patience': 10,
    'bwas_restart_stagnation': 8,
  },
  # Profile D — Heavy Graph / Dense Topologies (Large branching factors)
  'heavy_graph': {
    'ants': 150,
    'evaporation_rate': 0.15,
    'f_ini': 1.0,
    'f_min': 1e-6,
    'alfa': 2.0,
    'beta': 1.0,
    'epomax': 5000,
    'local_evaporation_rate': 0.05,
    'transition_probability': 0.70,
    'global_best_patience': 30,
    'bwas_restart_stagnation': 12,
  },
  # Profile E — Dynamic Network / Reactive (Fast adaptation to dynamic weights)
  'dynamic_network': {
    'ants': 50,
    'evaporation_rate': 0.60,
    'f_ini': 0.8,
    'f_min': 1e-5,
    'alfa': 1.0,
    'beta': 1.5,
    'epomax': 1500,
    'local_evaporation_rate': 0.30,
    'transition_probability': 0.40,
    'global_best_patience': 10,
    'bwas_restart_stagnation': 4,
  },
  # Profile F — Greedy Fast / Low-Latency Convergence
  'greedy_fast': {
    'ants': 30,
    'evaporation_rate': 0.50,
    'f_ini': 0.1,
    'f_min': 1e-6,
    'alfa': 0.5,
    'beta': 5.0,
    'epomax': 500,
    'local_evaporation_rate': 0.20,
    'transition_probability': 0.95,
    'global_best_patience': 8,
    'bwas_restart_stagnation': 10,
  },
  # Profile G — Best-Worst Ant System (BWAS Specialization)
  'bwas_aggressive': {
    'ants': 50,
    'evaporation_rate': 0.20,
    'f_ini': None,
    'f_min': 1e-5,
    'alfa': 1.0,
    'beta': 2.5,
    'epomax': 1500,
    'local_evaporation_rate': 0.10,
    'transition_probability': 0.50,
    'global_best_patience': 15,
    'bwas_restart_stagnation': 5,
    'worst_penalty_rate': 0.30,
    'mutation_probability': 0.08,
    'mutation_scale': 2.5,
  },
}


def load_profile(name: str) -> dict:
  """Return a copy of the base settings updated with a named preset.

  The returned dict includes keys used across ACO/ACS implementations:
  - ants, evaporation_rate, f_ini, alfa, beta, epomax,
    local_evaporation_rate, transition_probability, shared global-best patience,
    and BWAS restart.
  """
  base = dict(settings)
  profile = presets.get(name)
  if profile:
    base.update(profile)
  return base
