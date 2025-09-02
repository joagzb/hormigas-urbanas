import logging
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from scripts.utils.generators import (
    generate_random_graph,
    generate_bus_routes,
    generate_bus_graph,
    merge_bus_and_map_graph,
    generate_pheromone_map,
)

def test_generate_random_graph_structure():
    np.random.seed(42)
    size = 10
    graph = generate_random_graph(size, max_weight=10)

    # basic keys
    assert set(graph.keys()) == {"node_index", "connections", "weights"}
    assert graph["node_index"] == set(range(size))
    assert set(graph["connections"].keys()) == set(range(size))
    assert set(graph["weights"].keys()) == set(range(size))

    # every node has an entry (even if no neighbors)
    for i in range(size):
        assert i in graph["connections"]
        assert i in graph["weights"]
        conns = graph["connections"][i]
        w = graph["weights"][i]
        
        # lengths match
        assert isinstance(conns, list)
        assert isinstance(w, list)
        assert len(conns) == len(w)
        
        # neighbors are valid and not self connected
        assert all(0 <= v < size and v != i for v in conns)
        
        # neighbors are within window [i-2, i+2]
        assert all(max(0, i - 2) <= v < min(size, i + 3) for v in conns)
        
        # weights are positive
        assert all(1 <= wt <= 10 for wt in w)
        
        
def test_merge_bus_and_map_graph_adds_get_on_off_edges(monkeypatch):
    # base map with empty lists for bus nodes to be added later
    map_graph = {
        "node_index": {0, 1, 2},
        "connections": {0: [], 1: [], 2: []},
        "weights": {0: [], 1: [], 2: []},
    }

    # one bus graph with route 0->1->2 and bus nodes 1000..1002
    buses_graph = [{
        "name": "bus line 0",
        "route": [0, 1, 2],
        "node_bus_index": {1000, 1001, 1002},
        "connections": {1000: [1001], 1001: [1002], 1002: []},
        "weights": {1000: [3.0], 1001: [4.0], 1002: []},
        "stops": [(0, 1000), (1, 1001), (2, 1002)],
    }]

    # patch boarding/alighting costs
    import scripts.utils.generators as mod
    monkeypatch.setattr(mod, "calculate_bus_get_on_cost", lambda: 3.3)
    monkeypatch.setattr(mod, "calculate_bus_get_off_cost", lambda: 0.01)

    merged = merge_bus_and_map_graph(map_graph, buses_graph)

    # bus nodes are merged into node_index
    assert {1000, 1001, 1002}.issubset(merged["node_index"])

    # get-on edges from map node to bus node for all but last stop of route
    assert merged["connections"][0][-1] == 1000
    assert merged["weights"][0][-1] == 3.3
    assert merged["connections"][1][-1] == 1001
    assert merged["weights"][1][-1] == 3.3

    # get-off edges from bus node to map node for every stop
    assert merged["connections"][1000][-1] == 0
    assert merged["weights"][1000][-1] == 0.01
    assert merged["connections"][1001][-1] == 1
    assert merged["weights"][1001][-1] == 0.01
    assert merged["connections"][1002][-1] == 2
    assert merged["weights"][1002][-1] == 0.01

    # buses list preserved
    assert merged["buses"] == buses_graph
    
    
def test_generate_bus_graph_builds_bus_nodes_and_costs(monkeypatch):
    # deterministic small map graph with a known route 0->1->2
    map_graph = {
        "node_index": {0, 1, 2},
        "connections": {0: [1], 1: [2], 2: []},
        "weights": {0: [10.0], 1: [20.0], 2: []},
    }

    # patch generate_bus_routes to return one known route
    import scripts.utils.generators as mod
    monkeypatch.setattr(mod, "generate_bus_routes", lambda g: [{"name": "bus line 0", "route": [0, 1, 2]}])

    # patch cost function to a simple multiplier (e.g., 0.5x)
    monkeypatch.setattr(mod, "calculate_bus_time_travel_cost", lambda w: 0.5 * w)

    # patch get_connection_weight to read from our map_graph
    def fake_get_connection_weight(g, a, b):
        if b in g["connections"][a]:
            idx = g["connections"][a].index(b)
            return g["weights"][a][idx]
        return None
    monkeypatch.setattr(mod, "get_connection_weight", fake_get_connection_weight)

    buses_graph = generate_bus_graph(map_graph)

    assert len(buses_graph) == 1
    bus = buses_graph[0]
    # offset rule
    assert bus["node_bus_index"] == {1000, 1001, 1002}
    # route copied
    assert bus["route"] == [0, 1, 2]
    # connections from bus nodes follow route and use transformed weights
    assert bus["connections"][1000] == [1001]
    assert bus["connections"][1001] == [1002]
    assert bus["weights"][1000] == [5.0]    # 0.5 * 10
    assert bus["weights"][1001] == [10.0]   # 0.5 * 20
    # stops include a (map_node, bus_node) for each non-terminal step
    assert bus["stops"] == [(0, 1000), (1, 1001)]
    
    
def test_generate_pheromone_map_aligns_with_connections():
    map_graph = {
        "connections": {
            0: [1, 2],
            1: [2],
            2: [],
        }
    }
    initial = 0.5
    pher = generate_pheromone_map(map_graph, initial_lvl=initial)

    # same keys and lengths as connections; values start at initial
    assert set(pher.keys()) == {0, 1, 2}
    assert pher[0].shape == (2,)
    assert pher[1].shape == (1,)
    assert pher[2].shape == (0,)
    assert np.allclose(pher[0], initial)
    assert np.allclose(pher[1], initial)