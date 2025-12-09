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

    Base map nodes are treated as integers < 100000. Bus nodes are >= 100000 and
    are placed with a small offset next to their base node.
    """
    try:
        # Prefer explicit branching over one-liners for clarity
        node_index_obj = graph_dict.get("node_index")
        if isinstance(node_index_obj, (set, list)):
            nodes_sorted = sorted(node_index_obj)
        else:
            nodes_sorted = list(graph_nx.nodes)

        # Build base_nodes with a standard loop for clarity
        base_nodes: List[int] = []
        for n in nodes_sorted:
            if isinstance(n, int) and n < 100000:
                base_nodes.append(n)
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
        bus_nodes: List[int] = []
        for n in nodes_sorted:
            if isinstance(n, int) and n >= 100000:
                bus_nodes.append(n)
        for n in bus_nodes:
            base = n - 100000
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


def node_style(graph_nx: nx.DiGraph, base_size: int = 300, bus_size: int = 450) -> Tuple[List[str], List[int]]:
    """Per-node color and size sequences for drawing.

    Sizes are provided by caller to allow adaptive scaling.
    """
    colors = ["orange" if n >= 100000 else "lightblue" for n in graph_nx.nodes]
    sizes = [bus_size if n >= 100000 else base_size for n in graph_nx.nodes]
    return colors, sizes


def draw_base_graph(
    graph_nx: nx.DiGraph,
    positions: Dict[int, Tuple[float, float]],
    *,
    draw_labels: bool,
    node_size_base: int,
    node_size_bus: int,
    draw_edges: bool,
    edge_alpha: float,
    edge_width: float,
):
    """Draw nodes, optional labels and optional edges for the base graph."""
    colors, sizes = node_style(graph_nx, base_size=node_size_base, bus_size=node_size_bus)
    nx.draw_networkx_nodes(graph_nx, positions, node_color=colors, node_size=sizes, linewidths=0.2)

    if draw_labels:
        nx.draw_networkx_labels(
            graph_nx,
            positions,
            font_size=7,
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6),
            verticalalignment="center",
            horizontalalignment="center",
        )

    if draw_edges:
        nx.draw_networkx_edges(graph_nx, positions, edge_color="gray", alpha=edge_alpha, arrows=True, width=edge_width)


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

    # Adaptive figure size and density handling
    n_nodes = graph_nx.number_of_nodes()
    n_edges = graph_nx.number_of_edges()

    # Derive an approximate side length if positions come from a grid
    # Estimate grid side length using an explicit loop (avoid generator one-liner)
    base_count = 0
    for n in graph_nx.nodes:
        if isinstance(n, int) and n < 100000:
            base_count += 1
    side_guess = int(math.sqrt(max(1, base_count)))

    # Figure size scales with side length but is capped to avoid gigantic figures
    if side_guess >= 2:
        scale = max(6.0, min(22.0, side_guess * 0.35))
        figsize = (scale, scale)
    else:
        # fallback based on node count density
        scale = max(6.0, min(22.0, math.sqrt(max(1, n_nodes)) * 0.5))
        figsize = (scale, scale)

    plt.figure(figsize=figsize)

    # Decide rendering detail based on size thresholds
    if n_nodes <= 400:
        draw_labels = True
        node_size_base, node_size_bus = 300, 450
        draw_edges = True
        edge_alpha, edge_width = 0.6, 1.0
    elif n_nodes <= 5000:
        draw_labels = False
        node_size_base, node_size_bus = 40, 60
        draw_edges = True
        edge_alpha, edge_width = 0.35, 0.6
    elif n_nodes <= 20000:
        draw_labels = False
        node_size_base, node_size_bus = 8, 10
        # draw edges lightly; large graphs can be very heavy
        draw_edges = n_edges < 80000
        edge_alpha, edge_width = 0.15, 0.3
    else:
        draw_labels = False
        node_size_base, node_size_bus = 2, 3
        draw_edges = False  # too dense; show only nodes
        edge_alpha, edge_width = 0.05, 0.2

    draw_base_graph(
        graph_nx,
        positions,
        draw_labels=draw_labels,
        node_size_base=node_size_base,
        node_size_bus=node_size_bus,
        draw_edges=draw_edges,
        edge_alpha=edge_alpha,
        edge_width=edge_width,
    )
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
