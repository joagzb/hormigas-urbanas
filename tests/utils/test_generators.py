import logging
import numpy as np

from src.scripts.utils.generators import (
    generate_random_graph,
    generate_bus_routes,
    generate_bus_graph,
    merge_bus_and_map_graph,
    generate_pheromone_map,
)

def test_generate_random_graph_structure():
    """Produces a well-formed graph with valid neighbor lists and weights."""
    np.random.seed(42)
    size = 10
    generated_graph = generate_random_graph(size, max_weight=10)

    # Basic keys and node index coverage
    assert set(generated_graph.keys()) == {"node_index", "connections", "weights"}
    assert generated_graph["node_index"] == set(range(size))
    assert set(generated_graph["connections"].keys()) == set(range(size))
    assert set(generated_graph["weights"].keys()) == set(range(size))

    # Every node has an entry (even if no neighbors)
    for node_id in range(size):
        assert node_id in generated_graph["connections"]
        assert node_id in generated_graph["weights"]
        neighbors = generated_graph["connections"][node_id]
        weights = generated_graph["weights"][node_id]

        # Lengths match and are lists
        assert isinstance(neighbors, list)
        assert isinstance(weights, list)
        assert len(neighbors) == len(weights)

        # Valid neighbors, no self-connections, and within local window
        assert all(0 <= v < size and v != node_id for v in neighbors)
        assert all(max(0, node_id - 2) <= v < min(size, node_id + 3) for v in neighbors)

        # Positive weights within the specified bound
        assert all(1 <= wt <= 10 for wt in weights)
        
def test_merge_bus_and_map_graph_adds_get_on_off_edges(monkeypatch):
    """Merging adds get-on edges at non-terminal stops and get-off edges everywhere."""
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
    import src.scripts.utils.generators as mod
    monkeypatch.setattr(mod, "calculate_bus_get_on_cost", lambda: 3.3)
    monkeypatch.setattr(mod, "calculate_bus_get_off_cost", lambda: 0.01)

    merged = merge_bus_and_map_graph(map_graph, buses_graph)

    # Bus nodes are merged into node_index
    assert {1000, 1001, 1002}.issubset(merged["node_index"])

    # Get-on edges from map node to bus node for all but last stop of route
    assert merged["connections"][0][-1] == 1000
    assert merged["weights"][0][-1] == 3.3
    assert merged["connections"][1][-1] == 1001
    assert merged["weights"][1][-1] == 3.3

    # Get-off edges from bus node to map node for every stop
    assert merged["connections"][1000][-1] == 0
    assert merged["weights"][1000][-1] == 0.01
    assert merged["connections"][1001][-1] == 1
    assert merged["weights"][1001][-1] == 0.01
    assert merged["connections"][1002][-1] == 2
    assert merged["weights"][1002][-1] == 0.01

    # buses list preserved
    assert merged["buses"] == buses_graph
    
    
def test_generate_bus_graph_builds_bus_nodes_and_costs(monkeypatch):
    """Builds bus nodes following a known route with transformed costs."""
    # Deterministic small map graph with a known route 0->1->2
    map_graph = {
        "node_index": {0, 1, 2},
        "connections": {0: [1], 1: [2], 2: []},
        "weights": {0: [10.0], 1: [20.0], 2: []},
    }

    # patch generate_bus_routes to return one known route
    import src.scripts.utils.generators as mod
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
    # Offset rule (1000-based offset)
    assert bus["node_bus_index"] == {1000, 1001, 1002}
    # route copied
    assert bus["route"] == [0, 1, 2]
    # Connections from bus nodes follow route and use transformed weights
    assert bus["connections"][1000] == [1001]
    assert bus["connections"][1001] == [1002]
    assert bus["weights"][1000] == [5.0]    # 0.5 * 10
    assert bus["weights"][1001] == [10.0]   # 0.5 * 20
    # Stops include a (map_node, bus_node) for each non-terminal step
    assert bus["stops"] == [(0, 1000), (1, 1001)]
    
    
def test_generate_pheromone_map_aligns_with_connections():
    """Pheromone map mirrors connections and initializes with the given level."""
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
