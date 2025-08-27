import networkx as nx
import matplotlib.pyplot as plt


def draw_graph(graph, path=None):
    """Draw a graph highlighting bus nodes and optionally a specific path.

    Parameters
    ----------
    graph : dict
        Graph structure with ``node_index``, ``connections`` and ``weights``.
    path : list[int], optional
        Sequence of node IDs representing a path to highlight.
    """
    g = nx.DiGraph()

    # Add nodes
    for node in graph["node_index"]:
        g.add_node(node)

    # Add weighted edges
    for node, neighbors in graph["connections"].items():
        weights = graph["weights"].get(node, [])
        for neighbor, weight in zip(neighbors, weights):
            g.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(g)

    # Determine node colors: bus nodes are orange
    node_colors = ["orange" if n >= 1000 else "lightblue" for n in g.nodes]

    nx.draw_networkx_nodes(g, pos, node_color=node_colors)
    nx.draw_networkx_labels(g, pos)
    nx.draw_networkx_edges(g, pos, edge_color="gray", arrows=True)

    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(
            g,
            pos,
            edgelist=path_edges,
            edge_color="red",
            width=2,
            arrows=True,
        )

    plt.axis("off")
    plt.show()


def main():
    from .generators import generate_random_graph
    from .route_finder import dijkstra

    graph = generate_random_graph(10)
    example_path = dijkstra(graph, 0, 5)
    draw_graph(graph, example_path)


if __name__ == "__main__":
    main()
