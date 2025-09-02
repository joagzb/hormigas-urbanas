import logging
from math import isclose

from src.scripts.utils.weights import (
    get_connection_weight,
    haversine,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_get_connection_weight():
    graph = {
        "connections": {
            0: [1], 
            1: [], 
        },
        "weights": {
            0: [5.0], 
            1: [],
        },
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
    distance_1 = haversine(lat1, lon1, lat2, lon2)
    logger.info("Computed distance: %s", distance_1)
    
    # Buenos Aires to São Paulo ≈ 1674 km
    lat1, lon1 = -34.6037, -58.3816
    lat2, lon2 = -23.5505, -46.6333
    logger.info("Testing haversine BA → SP")
    logger.info(
        "Testing haversine with lat1=%s, lon1=%s, lat2=%s, lon2=%s",
        lat1,
        lon1,
        lat2,
        lon2,
    )
    distance_2 = haversine(lat1, lon1, lat2, lon2)
    logger.info("Computed distance: %s", distance_2)
    
    assert isclose(distance_1, 111.195, rel_tol=1e-3) and isclose(distance_2, 1674, rel_tol=1e-2)
