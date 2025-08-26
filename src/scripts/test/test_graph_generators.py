"""Utility generators for simple square graphs used in tests.

These helpers are primarily intended for quick experimentation and unit tests.
They build a deterministic square grid and a single vertical bus line to
validate routing algorithms.
"""

from ..utils.weights import calculate_bus_time_travel_cost
from ..utils.generators import merge_bus_and_map_graph
from ..utils.route_finder import dijkstra

def generate_square_city_graph(size, fixed_weight):
    """
    Generate a square city graph with the given size and fixed edge weight.

    Parameters:
        size (int): The size of the city (number of nodes per side).
        fixed_weight (float): The weight for each edge.

    Returns:
        dict: A dictionary representing the graph with nodes, connections, and weights.
    """
    graph = {
        "node_index": set(range(size * size)),
        "connections": [],
        "weights": [],
    }

    for i in range(size):
        for j in range(size):
            current_node = i * size + j

            graph["connections"].append((current_node, []))
            graph["weights"].append((current_node, []))

            if i > 0:  # North
                graph["connections"][-1][1].append(current_node - size)
                graph["weights"][-1][1].append(fixed_weight)

            if i < size - 1:  # South
                graph["connections"][-1][1].append(current_node + size)
                graph["weights"][-1][1].append(fixed_weight)

            if j > 0:  # West
                graph["connections"][-1][1].append(current_node - 1)
                graph["weights"][-1][1].append(fixed_weight)

            if j < size - 1:  # East
                graph["connections"][-1][1].append(current_node + 1)
                graph["weights"][-1][1].append(fixed_weight)

    graph["connections"] = dict(graph["connections"])
    graph["weights"] = dict(graph["weights"])

    return graph


def generate_bus_line_square_city(size, fixed_weight):
    """
    Generate a bus line graph in a square city with the given size and fixed edge weight.

    Parameters:
        size (int): The size of the city (number of nodes per side).
        fixed_weight (float): The weight for each edge.

    Returns:
        list: A list containing the bus line graph.
    """
    bus_node_index_offset = 1000
    distance = calculate_bus_time_travel_cost(fixed_weight)

    route = list(range(5, size * size, size))
    node_bus_index = {node + bus_node_index_offset for node in route}

    bus_dict = {
        "name": "bus line UNIQUE",
        "stops": [],
        "route": route,
        "node_bus_index": node_bus_index,
        "connections": [],
        "weights": [],
    }

    for i, current_node in enumerate(route):
        bus_current = current_node + bus_node_index_offset
        bus_dict["stops"].append((current_node, bus_current))

        if i < len(route) - 1:
            bus_next = route[i + 1] + bus_node_index_offset
            bus_dict["connections"].append((bus_current, [bus_next]))
            bus_dict["weights"].append((bus_current, [distance]))
        else:
            bus_dict["connections"].append((bus_current, []))
            bus_dict["weights"].append((bus_current, []))

    bus_dict["connections"] = dict(bus_dict["connections"])
    bus_dict["weights"] = dict(bus_dict["weights"])

    return [bus_dict]


def _compute_route_cost(graph, path):
    total = 0.0
    for start, end in zip(path, path[1:]):
        idx = graph["connections"][start].index(end)
        total += graph["weights"][start][idx]
    return total


if __name__ == "__main__":
    size = 10
    fixed_weight = 0.1
    map_graph = generate_square_city_graph(size, fixed_weight)
    buses_graph = generate_bus_line_square_city(size, fixed_weight)
    merged = merge_bus_and_map_graph(map_graph, buses_graph)

    route = dijkstra(merged, 5, 95)
    print("Example route:", route)
    print("Route cost:", _compute_route_cost(merged, route))

