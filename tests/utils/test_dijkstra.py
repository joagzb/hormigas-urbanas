from src.scripts.utils.route_finder import dijkstra


def test_dijkstra_finds_path_on_linear_chain():
    """Finds the only path on a simple 0->1->2->3 chain."""
    simple_graph = {
        "node_index": {0, 1, 2, 3},
        "connections": {0: [1], 1: [2], 2: [3], 3: []},
        "weights": {0: [1.0], 1: [1.0], 2: [1.0], 3: []},
    }

    path = dijkstra(simple_graph, 0, 3)
    expected_path = [0, 1, 2, 3]
    assert path == expected_path
