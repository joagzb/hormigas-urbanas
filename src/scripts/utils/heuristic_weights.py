import numpy as np

# Support both execution modes:
# - running from project root where `src` is top-level package
# - tests that insert `src` into sys.path (so configuration is top-level)
try:
    from src.configuration.graph_settings import settings as graph_settings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for direct test execution
    from configuration.graph_settings import settings as graph_settings

_BUS_EXIT_COST = graph_settings.get("bus_get_off", 0.0)
_BUS_TRAVEL_COST = graph_settings.get("bus_time_travel_cost", 0.1)
# Treat "free" exits as if they at least cost the time of continuing on the bus.
_BUS_EXIT_HEURISTIC = max(_BUS_TRAVEL_COST, _BUS_EXIT_COST)


def normalize_for_selection(weights: np.ndarray) -> np.ndarray:
    """Return a copy of weights with bus-exit edges floored for heuristic use.

    Bus-exit edges are modeled with an extremely small cost (wait time already
    paid when boarding) which causes the heuristic term (1 / weight)^beta to
    dominate selection and forces ants to disembark immediately. We keep the
    true cost intact for distance accumulation but floor the heuristic weight
    so transit stays attractive.
    """
    if _BUS_EXIT_COST <= 0:
        return weights

    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return arr

    mask = arr <= (_BUS_EXIT_COST + 1e-9)
    if not np.any(mask):
        return arr

    adjusted = arr.copy()
    adjusted[mask] = _BUS_EXIT_HEURISTIC
    return adjusted
