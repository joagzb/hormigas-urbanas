from src.scripts.utils.graph_visualizer import draw_graph


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
