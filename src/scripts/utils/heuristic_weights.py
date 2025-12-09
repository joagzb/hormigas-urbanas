import numpy as np

# Support both execution modes:
# - running from project root where `src` is top-level package
# - tests that insert `src` into sys.path (so configuration is top-level)
try:
    from src.configuration.graph_settings import settings as graph_settings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from configuration.graph_settings import settings as graph_settings

_BUS_EXIT_COST = graph_settings.get("bus_get_off", 0.0)
_BUS_TRAVEL_COST = graph_settings.get("bus_time_travel_cost", 0.1)
_BUS_BOARD_COST = graph_settings.get("wait_for_bus_cost", 0.0) + graph_settings.get("pay_for_bus_cost", 0.0)

# Heuristic surrogate weights used only for neighbor selection.
# Lower effective weight => edge is more attractive in the (1 / weight)^beta term.
# We bias ants to:
#   - prefer boarding over walking,
#   - strongly prefer staying on the bus over exiting early.
_BUS_BOARD_HEURISTIC = min(_BUS_BOARD_COST, _BUS_TRAVEL_COST)
_BUS_EXIT_HEURISTIC = max(_BUS_TRAVEL_COST * 4.0, _BUS_BOARD_COST + _BUS_TRAVEL_COST)


def normalize_for_selection(weights: np.ndarray) -> np.ndarray:
    """Return a copy of weights with bus board/exit edges adjusted for heuristic use.

    The true costs (used for distance accumulation) stay untouched. For
    neighbor selection, boarding edges are made competitively cheap and
    exit edges more expensive so ants tend to remain on the bus longer
    instead of exiting at the first opportunity.
    """
    if _BUS_EXIT_COST <= 0:
        return weights

    arr = np.asarray(weights, dtype=float)
    if arr.size == 0:
        return arr

    adjusted = arr.copy()

    # Make bus-exit edges heuristically expensive to discourage early exits.
    exit_mask = _BUS_EXIT_COST > 0 and (arr <= (_BUS_EXIT_COST + 1e-9))
    if np.any(exit_mask):
        adjusted[exit_mask] = _BUS_EXIT_HEURISTIC

    # Make boarding edges heuristically cheap to encourage using the bus.
    board_mask = _BUS_BOARD_COST > 0 and np.isclose(arr, _BUS_BOARD_COST)
    if np.any(board_mask):
        adjusted[board_mask] = min(_BUS_BOARD_HEURISTIC, _BUS_BOARD_COST)

    return adjusted
