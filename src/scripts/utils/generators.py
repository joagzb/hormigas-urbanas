import numpy as np
from .weights import (
    get_connection_weight,
    calculate_bus_time_travel_cost,
    calculate_bus_get_on_cost,
    calculate_bus_get_off_cost,
)
from .route_finder import dijkstra

def generate_random_graph(size, max_weight=100):
    """Generate a random graph with connections between nearby nodes.

    The original implementation skipped nodes ``0`` and ``1`` when building
    connections, which produced graphs with missing entries. This version
    ensures that every node has an entry in the ``connections`` and ``weights``
    dictionaries and limits neighbor lookups to valid node indices.

    Parameters
    ----------
    size: int
        Number of nodes in the graph.
    max_weight: float, optional
        Maximum weight for any edge.

    Returns
    -------
    dict
        Random graph dictionary with node indices, connections and weights.
    """

    graph = {
        "node_index": set(range(size)),
        "connections": [],
        "weights": [],
    }

    for current_node in range(size):
        graph["connections"].append((current_node, []))
        graph["weights"].append((current_node, []))

        for next_node in range(max(0, current_node - 2), min(size, current_node + 3)):
            if current_node == next_node:
                continue

            if np.random.random() > 0.3:  # 70% chance to have a connection
                distance = np.random.randint(1, max_weight + 1)
                graph["connections"][-1][1].append(next_node)
                graph["weights"][-1][1].append(distance)

    graph["connections"] = dict(graph["connections"])
    graph["weights"] = dict(graph["weights"])

    return graph


def generate_bus_graph(graph):
    bus_routes = generate_bus_routes(graph)
    buses_graph = []

    for route in bus_routes:
        bus_node_index_offset = 1000 + (1000 * len(buses_graph))

        bus_dict = {
            'name': route['name'],
            'stops': [],  # Connections between map nodes and bus nodes
            'route': route['route'],
            'node_bus_index': {node + bus_node_index_offset for node in route['route']},
            'connections': [],
            'weights': []
        }

        for i in range(len(route["route"]) - 1):
            current_node = route["route"][i]
            next_node = route["route"][i + 1]
            bus_current_node = current_node + bus_node_index_offset
            bus_next_node = next_node + bus_node_index_offset
            original_weight = get_connection_weight(graph, current_node, next_node)
            bus_dict["stops"].append((current_node, bus_current_node))

            if original_weight is None:
                # Still ensure the bus node exists in the adjacency map
                bus_dict["connections"].append((bus_current_node, []))
                bus_dict["weights"].append((bus_current_node, []))
                continue

            cost = calculate_bus_time_travel_cost(original_weight)
            bus_dict["connections"].append((bus_current_node, [bus_next_node]))
            bus_dict["weights"].append((bus_current_node, [cost]))

        # Note: Do NOT add the terminal stop to 'stops'; boarding should only
        # be possible at non-terminal stops. Tests expect this behavior.

        bus_dict['connections'] = dict(bus_dict['connections'])
        bus_dict['weights'] = dict(bus_dict['weights'])
        buses_graph.append(bus_dict)

    return buses_graph



def generate_bus_routes(graph):
    """
    Adds bus routes to the graph.

    Parameters:
    graph (dict): The graph dictionary with node indices, connections, and weights.

    Returns:
    list: List of bus routes. Each bus route is a dict with 'name' and 'route'.
    """
    bus_routes = []
    node_index_list = list(graph['node_index'])
    num_buses = max(1, int(np.round(len(node_index_list) / 20)))  # Add a bus for every 20 nodes, ensure at least one bus

    for bus_number in range(num_buses):
        start_node = np.random.choice(node_index_list)
        end_node = np.random.choice(node_index_list)

        while end_node == start_node:
            end_node = np.random.choice(node_index_list)

        route = dijkstra(graph, start_node, end_node)

        while route is None:
            end_node = np.random.choice(node_index_list)
            route = dijkstra(graph, start_node, end_node)

        bus_routes.append({
            "name": f"bus line {bus_number}",
            "route": route
        })

    return bus_routes


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
