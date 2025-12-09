settings = {
  "ants": 50,
  "f_ini": 0.2,  # initial level of pheromone
  "f_max": 0.9,  # max level of pheromone allowed for each edge (Max-Min ant)
  "f_min": 0.01, # min level of pheromone allowed for each edge (Best-worst ant and Max-Min ant)
  "evaporation_rate": 0.1,  # (p) pheromone evaporation level - lowered to improve memory retention
  "epomax": 500,
  "local_evaporation_rate": 0.1,  # rho - lowered to be consistent
  "transition_probability": 0.2,  # q0
  "alfa": 1.0,  # heuristic_weight (exponent on pheromone) - increased slightly to rely more on history
  "beta": 0.5,  # pheromone_weight (exponent on 1/cost) - decreased to reduce greediness against bus entry
}

presets = {
  # Profile A — bus‑friendly balanced
  "bus_friendly": {
    "ants": 80,
    "evaporation_rate": 0.35,
    "f_ini": 0.5,
    "alfa": 1.0,
    "beta": 0.35,
    "epomax": 2000,
    "local_evaporation_rate": 0.12,
    "transition_probability": 0.15,
  },
  # Profile B — stronger exploration early
  "explore_strong": {
    "ants": 120,
    "evaporation_rate": 0.25,
    "f_ini": 0.6,
    "alfa": 1.2,
    "beta": 0.25,
    "epomax": 3000,
    "local_evaporation_rate": 0.10,
    "transition_probability": 0.10,
  },
}

def load_profile(name: str) -> dict:
  """Return a copy of the base settings updated with a named preset.

  The returned dict includes keys used across ACO/ACS implementations:
  - ants, evaporation_rate, f_ini, alfa, beta, epomax,
    local_evaporation_rate, transition_probability
  """
  base = dict(settings)
  profile = presets.get(name)
  if profile:
    base.update(profile)
  return base
