import numpy as np
from .weights import *
from .route_finder import dijkstra

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
        'node_index': set(range(size)),
        'connections': [],
        'weights': []
    }

    for current_node in range(2,size):
        graph['connections'].append((current_node, list()))
        graph['weights'].append((current_node, list()))
        for next_node in range(current_node-2,current_node+3):
            if current_node != next_node:
                # Randomly decide the distance between nodes (or no connection)
                distance = np.random.randint(1, max_weight + 1) if np.random.random() > 0.3 else None  # 70% chance to have a connection
                if distance is not None:
                    graph['connections'][-1][1].append(next_node)
                    graph['weights'][-1][1].append(distance)

    graph['connections'] = dict(graph['connections'])
    graph['weights'] = dict(graph['weights'])

    return graph


def generate_bus_graph(graph):
    bus_routes = generate_bus_routes(graph)
    buses_graph = []

    for route in bus_routes:
        bus_node_index_offset = 1000 + (1000*len(buses_graph))

        bus_dict = {
            'name': route['name'],
            'stops': [],  # Connections between map nodes and bus nodes
            'route': route['route'],
            'node_bus_index': set(np.array(route['route']) + bus_node_index_offset),
            'connections': [],
            'weights': []
        }

        for i in range(len(route["route"]) - 1):
            current_node = route["route"][i]
            next_node = route["route"][i + 1]
            bus_current_node = current_node + bus_node_index_offset
            bus_next_node = next_node + bus_node_index_offset
            original_weight = get_connection_weight(graph, current_node, next_node)
            bus_dict['stops'].append((current_node, current_node + bus_node_index_offset))

            if original_weight is not None:
                cost = calculate_bus_time_travel_cost(original_weight)
                bus_dict['connections'].append((bus_current_node, list([bus_next_node])))
                bus_dict['weights'].append((bus_current_node, list([cost])))

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

        # Add bus stops to get on
        for i in range(len(bus_graph["route"]) - 1):
            route_weight = get_connection_weight(map_graph, bus_graph["route"][i], bus_graph["route"][i + 1])
            cost_get_on = calculate_bus_get_on_cost(route_weight)

            # Add connections and weights for getting on the bus
            start_map_node, start_bus_node = bus_graph["stops"][i]
            map_graph["connections"][start_map_node].append(start_bus_node)
            map_graph["weights"][start_map_node].append(cost_get_on)

        # Add bus stops to get off
        for i in range(len(bus_graph["route"])):
            cost_get_off = calculate_bus_get_off_cost()

            # Add connections and weights for getting off the bus
            start_map_node, start_bus_node = bus_graph["stops"][i]
            map_graph["connections"][start_bus_node].append(start_map_node)
            map_graph["weights"][start_bus_node].append(cost_get_off)

    # Add buses graph to the map graph
    map_graph["buses"] = buses_graph

    return map_graph


def generate_pheromone_map(map_graph, initial_lvl):
    pheromone_path = []

    for node,connection in map_graph["connections"].items():
        pheromone_path.append([node, initial_lvl + np.zeros(len(connection))])

    return dict(pheromone_path)