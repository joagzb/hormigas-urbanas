# Static generalized-cost coefficients; they are not real minutes or fares.
settings = {
  'wait_for_bus_cost': 0.9,
  'pay_for_bus_cost': 0.5,
  'bus_time_travel_cost': 0.3,  # 0.1 <= x <= 0.8
  'bus_get_off': 0.01,
}

# Preset bundles to tune bus attractiveness
graph_presets = {
  # Slightly more attractive bus: lower boarding costs and faster bus
  'bus_friendly': {
    'wait_for_bus_cost': 0.9,
    'pay_for_bus_cost': 0.5,
    'bus_time_travel_cost': 0.3,  # Recommended 0.1–0.8
  },
  # Strong bus preference: board is cheap, bus is much faster
  'bus_strong': {'wait_for_bus_cost': 0.5, 'pay_for_bus_cost': 0.3, 'bus_time_travel_cost': 0.2},
}


def load_graph_profile(name: str) -> dict:
  """Return a preset setting if exists, otherwise a default config."""
  base = dict(settings)
  preset = graph_presets.get(name)
  if preset:
    base.update(preset)
  return base
