import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from configuration import algorithm_settings, graph_settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_algorithm_settings_keys_and_values():
    expected_keys = {
        "ants",
        "f_ini",
        "f_max",
        "f_min",
        "evaporation_rate",
        "epomax",
        "local_evaporation_rate",
        "transition_probability",
        "alfa",
        "beta",
    }
    assert set(algorithm_settings.settings.keys()) == expected_keys

    rate_keys = {
        "f_ini",
        "f_max",
        "f_min",
        "evaporation_rate",
        "local_evaporation_rate",
        "transition_probability",
    }
    for key, value in algorithm_settings.settings.items():
        logger.info("algorithm_settings[%s] = %s", key, value)
        if key in rate_keys:
            assert 0 <= value <= 1
        elif key in {"ants", "epomax"}:
            assert value > 0
        else:
            assert value > 0


def test_graph_settings_keys_and_values():
    expected_keys = {
        "init_node",
        "final_node",
        "wait_for_bus_cost",
        "pay_for_bus_cost",
        "bus_time_travel_cost",
        "bus_get_off",
    }
    assert set(graph_settings.settings.keys()) == expected_keys

    rate_keys = {"bus_get_off"}
    for key, value in graph_settings.settings.items():
        logger.info("graph_settings[%s] = %s", key, value)
        if key in rate_keys:
            assert 0 < value <= 1
        else:
            assert value > 0

