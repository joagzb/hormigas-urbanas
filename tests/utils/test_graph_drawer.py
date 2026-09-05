import pytest

from src.scripts.utils.graph_visualizer import build_graph_from_dict, draw_graph, node_style


def test_draw_graph_saves_image(tmp_path):
    """Saves an image file when a save_path is provided."""
    small_graph = {
        "node_index": {0, 1, 2},
        "connections": {0: [1], 1: [2], 2: []},
        "weights": {0: [2.0], 1: [3.0], 2: []},
    }
    output_path = tmp_path / "graph.png"
    path_to_highlight = [0, 1, 2]

    draw_graph(small_graph, path=path_to_highlight, save_path=str(output_path))

    assert output_path.exists() and output_path.stat().st_size > 0


def test_bus_node_style_comes_from_service_metadata():
    bus_node = "bus:line:outbound:0"
    graph = {
        "node_index": {0, bus_node},
        "connections": {0: [bus_node], bus_node: [0]},
        "weights": {0: [1.4], bus_node: [0.01]},
        "edge_types": {0: ["board"], bus_node: ["alight"]},
        "buses": [{"line_id": "line", "direction": "outbound", "stops": [(0, bus_node)]}],
    }

    graph_nx = build_graph_from_dict(graph)
    colors, sizes = node_style(graph_nx)
    style = {node: (color, size) for node, color, size in zip(graph_nx.nodes, colors, sizes)}

    assert style[bus_node] == ("orange", 450)
    assert style[0] == ("lightblue", 300)


def test_build_graph_rejects_misaligned_edge_types():
    graph = {
        "node_index": {0, 1},
        "connections": {0: [1], 1: []},
        "weights": {0: [1.0], 1: []},
        "edge_types": {0: [], 1: []},
    }

    with pytest.raises(ValueError, match="connections and edge_types are not aligned"):
        build_graph_from_dict(graph)
