settings = {
  'ants': 20,
  'f_ini': None,  # None for automatic tau0, or a positive finite value.
  'f_min': 1e-6,  # Positive finite BWAS floor. Initialization and restarts clamp tau0 to this value.
  'evaporation_rate': 0.1,  # Global evaporation rate in [0, 1].
  'epomax': 100,  # Positive integer epoch limit.
  'local_evaporation_rate': 0.1,  # ACS local evaporation rate in [0, 1].
  'transition_probability': 0.9,  # ACS-only exploitation probability in [0, 1].
  'alfa': 1.0,  # Finite, non-negative pheromone exponent.
  'beta': 2.0,  # Finite, non-negative inverse-cost exponent.
  'global_best_patience': 10,
  'bwas_restart_stagnation': 8,  # Zero disables BWAS restarts; when enabled, use a positive value below global_best_patience.
  'worst_penalty_rate': 0.30,  # BWAS Extra worst-route evaporation rate in [0, 1].
  'mutation_probability': 0.08,  # BWAS Per-row mutation probability in [0, 1].
  'mutation_scale': 2.5,  # BWAS Finite, non-negative mutation magnitude multiplier.
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
    'bwas_restart_stagnation': 4,
  },
  # Profile G — ACS on large multimodal city.
  'acs_multimodal': {
    'ants': 80,
    'evaporation_rate': 0.20,
    'f_ini': None,
    'alfa': 1.0,
    'beta': 0.5,
    'epomax': 1500,
    'local_evaporation_rate': 0.10,
    'transition_probability': 0.15,
    'global_best_patience': 30,
  },
  # Profile H — Best-Worst Ant System (BWAS specialization).
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
    'global_best_patience': 30,
    'bwas_restart_stagnation': 5,
    'worst_penalty_rate': 0.30,  # Extra worst-route evaporation rate in [0, 1].
    'mutation_probability': 0.08,  # Per-row mutation probability in [0, 1].
    'mutation_scale': 2.5,  # Finite, non-negative mutation magnitude multiplier.
  },
}


def load_profile(name: str) -> dict:
  """Return a preset setting if exists, otherwise a default config."""
  base = dict(settings)
  profile = presets.get(name)
  if profile:
    base.update(profile)
  return base
