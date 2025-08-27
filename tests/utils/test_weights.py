import logging
from math import isclose

from src.scripts.utils.weights import (
    get_connection_weight,
    haversine,
    calculate_bus_get_on_cost,
    calculate_bus_get_off_cost,
    calculate_bus_time_travel_cost,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_get_connection_weight():
    graph = {
        "connections": {0: [1], 1: [0]},
        "weights": {0: [5.0], 1: [5.0]},
    }
    start_node, end_node = 0, 1
    logger.info(
        "Testing get_connection_weight with graph=%s, start_node=%s, end_node=%s",
        graph,
        start_node,
        end_node,
    )
    weight = get_connection_weight(graph, start_node, end_node)
    logger.info("Computed weight: %s", weight)
    assert weight == 5.0


def test_haversine():
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 0.0, 1.0
    logger.info(
        "Testing haversine with lat1=%s, lon1=%s, lat2=%s, lon2=%s",
        lat1,
        lon1,
        lat2,
        lon2,
    )
    distance = haversine(lat1, lon1, lat2, lon2)
    logger.info("Computed distance: %s", distance)
    assert isclose(distance, 111.195, rel_tol=1e-3)


def test_calculate_bus_get_on_cost():
    logger.info("Testing calculate_bus_get_on_cost with settings")
    cost = calculate_bus_get_on_cost()
    logger.info("Computed get-on cost: %s", cost)
    assert cost == 3.3


def test_calculate_bus_get_off_cost():
    logger.info("Testing calculate_bus_get_off_cost with settings")
    cost = calculate_bus_get_off_cost()
    logger.info("Computed get-off cost: %s", cost)
    assert cost == 0.01


def test_calculate_bus_time_travel_cost():
    route_weight = 10.0
    logger.info("Testing calculate_bus_time_travel_cost with route_weight=%s", route_weight)
    cost = calculate_bus_time_travel_cost(route_weight)
    logger.info("Computed time travel cost: %s", cost)
    assert cost == 5.0
