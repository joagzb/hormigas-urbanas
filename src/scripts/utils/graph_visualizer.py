import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx


def build_graph_from_dict(graph_dict: dict) -> nx.DiGraph:
    """Create a NetworkX DiGraph from a simple graph dictionary.

    Expected keys in graph_dict: "node_index", "connections", "weights".
    """
    graph_nx = nx.DiGraph()

    for node in graph_dict["node_index"]:
        graph_nx.add_node(node)

    for node, neighbors in graph_dict["connections"].items():
        weights = graph_dict["weights"].get(node, [])
        for neighbor, weight in zip(neighbors, weights):
            graph_nx.add_edge(node, neighbor, weight=weight)

    return graph_nx


def _grid_positions(graph_dict: dict, graph_nx: nx.DiGraph) -> Optional[Dict[int, Tuple[float, float]]]:
    """Return a grid layout when base nodes are 0..n*n-1; otherwise None.

    Base map nodes are treated as integers < 1000. Bus nodes are >= 1000 and
    are placed with a small offset next to their base node.
    """
    try:
        nodes_sorted = (
            sorted(graph_dict["node_index"]) if isinstance(graph_dict.get("node_index"), (set, list)) else list(graph_nx.nodes)
        )

        base_nodes = [n for n in nodes_sorted if isinstance(n, int) and n < 1000]
        if not base_nodes:
            return None

        max_base = max(base_nodes)
        if set(base_nodes) != set(range(max_base + 1)):
            return None

        side = int(math.isqrt(max_base + 1))
        if side * side != (max_base + 1):
            return None

        positions: Dict[int, Tuple[float, float]] = {}
        # Place base nodes on a grid; row-major indexing
        for n in base_nodes:
            row, col = divmod(n, side)
            positions[n] = (col, -row)

        # Place bus nodes slightly offset from their base station
        bus_nodes = [n for n in nodes_sorted if isinstance(n, int) and n >= 1000]
        for n in bus_nodes:
            base = n - 1000
            if base in positions:
                x, y = positions[base]
                positions[n] = (x + 0.25, y)
            else:
                positions[n] = (0.0, 0.0)

        return positions
    except Exception:
        return None


def compute_positions(graph_dict: dict, graph_nx: nx.DiGraph) -> Dict[int, Tuple[float, float]]:
    """Compute node positions with grid preference and spring fallback."""
    return _grid_positions(graph_dict, graph_nx) or nx.spring_layout(graph_nx, seed=42)


def node_style(graph_nx: nx.DiGraph) -> Tuple[List[str], List[int]]:
    """Per-node color and size sequences for drawing."""
    colors = ["orange" if n >= 1000 else "lightblue" for n in graph_nx.nodes]
    sizes = [450 if n >= 1000 else 300 for n in graph_nx.nodes]
    return colors, sizes


def draw_base_graph(graph_nx: nx.DiGraph, positions: Dict[int, Tuple[float, float]]):
    """Draw nodes, labels and edges for the base graph."""
    colors, sizes = node_style(graph_nx)
    nx.draw_networkx_nodes(graph_nx, positions, node_color=colors, node_size=sizes, linewidths=0.5)
    nx.draw_networkx_labels(
        graph_nx,
        positions,
        font_size=7,
        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
        verticalalignment="center",
        horizontalalignment="center",
    )
    nx.draw_networkx_edges(graph_nx, positions, edge_color="gray", alpha=0.6, arrows=True, width=1)


def highlight_route(graph_nx: nx.DiGraph, positions: Dict[int, Tuple[float, float]], path: List[int]):
    """Highlight a route on top of an existing graph drawing."""
    if not path:
        return
    path_edges = list(zip(path[:-1], path[1:]))
    nx.draw_networkx_edges(
        graph_nx,
        positions,
        edgelist=path_edges,
        edge_color="red",
        width=2,
        arrows=True,
    )


def draw_graph(graph: dict, path: Optional[List[int]] = None, save_path: Optional[str] = None):
    """Draw a graph highlighting bus nodes and optionally a specific path.

    Parameters
    ----------
    graph : dict
        Graph structure with "node_index", "connections" and "weights".
    path : list[int], optional
        Sequence of node IDs representing a path to highlight.
    save_path : str, optional
        If provided, saves the figure to this path; otherwise shows it.
    """
    graph_nx = build_graph_from_dict(graph)
    positions = compute_positions(graph, graph_nx)

    draw_base_graph(graph_nx, positions)
    if path:
        highlight_route(graph_nx, positions, path)

    plt.axis("off")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Graph saved to {save_path}")
    else:
        plt.show()

