import numpy as np
from .weights import (
    get_connection_weight,
    calculate_bus_time_travel_cost,
    calculate_bus_get_on_cost,
    calculate_bus_get_off_cost,
)
from .route_finder import dijkstra

def merge_bus_and_map_graph(map_graph, buses_graph):
    """
    Merge bus graph into the map graph.

    Parameters:
        map_graph (dict): The main graph representing the city.
        buses_graph (list): A list of bus graphs to be merged into the main graph.

    Returns:
        dict: The merged graph with both map and bus nodes.
    """
    for bus_graph in buses_graph:
        # Update map graph with bus nodes and connections
        map_graph["node_index"].update(bus_graph["node_bus_index"])
        map_graph["connections"].update(bus_graph["connections"])
        map_graph["weights"].update(bus_graph["weights"])

        route = bus_graph["route"]
        for i, (start_map_node, start_bus_node) in enumerate(bus_graph["stops"]):
            if i < len(route) - 1:
                cost_get_on = calculate_bus_get_on_cost()
                map_graph["connections"].setdefault(start_map_node, [])
                map_graph["weights"].setdefault(start_map_node, [])
                map_graph["connections"][start_map_node].append(start_bus_node)
                map_graph["weights"][start_map_node].append(cost_get_on)

            cost_get_off = calculate_bus_get_off_cost()
            map_graph["connections"].setdefault(start_bus_node, [])
            map_graph["weights"].setdefault(start_bus_node, [])
            map_graph["connections"][start_bus_node].append(start_map_node)
            map_graph["weights"][start_bus_node].append(cost_get_off)

    # Add buses graph to the map graph
    map_graph["buses"] = buses_graph

    return map_graph


def generate_pheromone_map(map_graph, initial_lvl):
    """Create a pheromone map aligned with the graph connections."""

    pheromone_path = {}
    for node, connections in map_graph["connections"].items():
        pheromone_path[node] = initial_lvl + np.zeros(len(connections))

    return pheromone_path
