import numpy as np
from .weights import get_connection_weight
from .route_finder import dijkstra
from configuration.graph_settings import settings

def get_adjacency_matrix_from_graph(graph):
    """
    Generates an adjacency matrix from a graph dictionary.

    Parameters:
    graph (dict): A dictionary representing the graph with 'node_index', 'connections', and 'weights'.

    Returns:
    np.ndarray: Adjacency matrix representation of the graph.
    """
    size = len(graph['node_index'])
    adjacency_matrix = np.zeros((size, size), dtype=float)

    for connection, weight in zip(graph['connections'], graph['weights']):
        adjacency_matrix[connection[0], connection[1]] = weight

    return adjacency_matrix


def generate_random_graph(size, max_weight=100):
    """
    Generates a random graph dictionary for a graph of given size.

    Parameters:
    size (int): Number of nodes in the graph.
    max_weight (float): Maximum weight for any edge.

    Returns:
    dict: Random graph dictionary with node indices, connections, and weights.
    """
    graph = {
        'node_index': list(range(size)),
        'connections': [],
        'weights': []
    }

    for i in range(size):
        for j in range(size):
            if i != j:
                # Randomly decide the distance between nodes (or no connection)
                distance = np.random.randint(1, max_weight + 1) if np.random.random() > 0.4 else None  # 60% chance to have a connection
                if distance is not None:
                    graph['connections'].append((i, j))
                    graph['weights'].append(distance)

    return graph


def generate_bus_graph(graph):
    buses_routes=generate_bus_routes(graph)
    buses_graph=[]

    for route in buses_routes:

        bus_dict = {
            'name': route['name'],
            'node_index': route['route'],
            'connections': [],
            'weights': []
        }

        for i in range(len(route["route"]) - 1):
            current_node = route["route"][i]
            next_node = route["route"][i + 1]
            original_weight = get_connection_weight(graph, current_node, next_node)

            if original_weight is not None:
                if i == 0:
                    cost_get_on = (original_weight * settings["wait_for_bus_cost"] *
                                   settings["pay_for_bus_cost"] * settings["bus_time_travel_cost"])
                    bus_dict['connections'].append((current_node, current_node))
                    bus_dict['weights'].append(cost_get_on)

                cost = original_weight * settings["bus_time_travel_cost"]
                bus_dict['connections'].append((current_node, next_node))
                bus_dict['weights'].append(cost)

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

    num_buses = max(1, int(np.round(len(graph['node_index']) / 20)))  # Add a bus for every 20 nodes, ensure at least one bus

    for bus_number in range(num_buses):
        start_node = np.random.choice(graph['node_index'])
        end_node = np.random.choice(graph['node_index'])
        while end_node == start_node:
            end_node = np.random.choice(graph['node_index'])

        route = dijkstra(graph, start_node, end_node)
        bus_routes.append({
            "name": f"bus line {bus_number}",
            "route": route
        })

    return bus_routes


def merge_bus_and_map_graph(map_graph,buses_graph):
    for bus_graph in buses_graph:
        map_graph["node_index"].extend(bus_graph["node_bus_index"])
        map_graph["connections"].extend(bus_graph["connections"])
        map_graph["weights"].extend(bus_graph["weights"])
    map_graph["buses"] = buses_graph