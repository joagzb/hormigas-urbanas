import numpy as np

# Support both execution modes:
# - running from project root where `src` is top-level package
# - tests that insert `src` into sys.path (so configuration is top-level)
try:
    from src.configuration.graph_settings import settings as graph_settings  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from configuration.graph_settings import settings as graph_settings

def normalize_for_selection(weights: np.ndarray, edge_types: np.ndarray) -> np.ndarray:
    """Return a copy of weights with bus board/exit edges adjusted for heuristic use.

    The true costs (used for distance accumulation) stay untouched. For
    neighbor selection, boarding edges are made competitively cheap and
    exit edges more expensive so ants tend to remain on the bus longer
    instead of exiting at the first opportunity.
    """
    arr = np.asarray(weights, dtype=float)
    types = np.asarray(edge_types, dtype=object)
    if arr.shape != types.shape:
        raise ValueError("Weights and edge_types must be aligned")
    if arr.size == 0:
        return arr

    adjusted = arr.copy()
    bus_travel_cost = graph_settings.get("bus_time_travel_cost", 0.1)
    bus_board_cost = graph_settings.get(
        "wait_for_bus_cost", 0.0
    ) + graph_settings.get("pay_for_bus_cost", 0.0)
    bus_board_heuristic = min(bus_board_cost, bus_travel_cost)
    bus_exit_heuristic = max(bus_travel_cost * 4.0, bus_board_cost + bus_travel_cost)

    # Make bus-exit edges heuristically expensive to discourage early exits.
    exit_mask = types == "alight"
    if np.any(exit_mask):
        adjusted[exit_mask] = bus_exit_heuristic

    # Make boarding edges heuristically cheap to encourage using the bus.
    board_mask = types == "board"
    if np.any(board_mask):
        adjusted[board_mask] = bus_board_heuristic

    # Zero-cost edges remain zero for route accounting, but inverse-cost
    # selection always needs a positive denominator.
    adjusted[adjusted <= 0] = np.finfo(float).eps

    return adjusted
