"""Utility generators for simple square graphs used in tests.

These helpers are primarily intended for quick experimentation and unit tests.
They build a deterministic square grid and a single vertical bus line to
validate routing algorithms.
"""

from ..utils.graph_visualizer import draw_graph
import copy
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
    node_bus_index = set()
    for node in route:
        node_bus_index.add(node + bus_node_index_offset)

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
    fixed_weight = 1

    map_graph = generate_square_city_graph(size, fixed_weight)
    buses_graph = generate_bus_line_square_city(size, fixed_weight)
    full_graph = merge_bus_and_map_graph(copy.deepcopy(map_graph), buses_graph)

    # Prompt user for start and end nodes
    min_node, max_node = 0, size * size - 1

    def _prompt_node(prompt_text: str, default: int) -> int:
        while True:
            raw = input(f"{prompt_text} [{default}] (min {min_node}, max {max_node}): ").strip()
            if raw == "":
                return default
            try:
                val = int(raw)
                if min_node <= val <= max_node:
                    return val
                else:
                    print(f"Please enter a value between {min_node} and {max_node}.")
            except ValueError:
                print("Please enter a valid integer.")

    default_start, default_end = 3, 69
    start_node = _prompt_node("Enter start node", default_start)
    end_node = _prompt_node("Enter end node", default_end)

    route_solution = dijkstra(full_graph, start_node, end_node)
    route_solution_only_walking = dijkstra(map_graph, start_node, end_node)

    draw_graph(full_graph, route_solution, save_path="toy_city_graph_solution.png")
    draw_graph(map_graph, route_solution_only_walking, save_path="toy_city_graph_solution_walking.png")

    print("Route solution:", route_solution)
    print("Route cost:", _compute_route_cost(full_graph, route_solution))
    print("Route solution (walking):", route_solution_only_walking)
    print("Route cost (walking):", _compute_route_cost(map_graph, route_solution_only_walking))
    
    if route_solution_only_walking >= route_solution:
        print("you'd better go by foot.")
    else:
        print("you'd better take the bus instead of walking.")
