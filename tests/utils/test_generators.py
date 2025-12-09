import numpy as np

from src.scripts.utils.generators import (
    merge_bus_and_map_graph,
    generate_pheromone_map,
)

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
