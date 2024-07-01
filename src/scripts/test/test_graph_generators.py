import numpy as np
from configuration.graph_settings import settings


def generate_square_city_graph():
    size=10
    fixed_weight=0.1

    graph = {
        'node_index': list(range(size * size)),
        'connections': [],
        'weights': []
    }

    for i in range(size):
        for j in range(size):
            current_node = i * size + j
            if i > 0:  # connect to the node North
                graph['connections'].append((current_node, current_node - size))
                graph['weights'].append(fixed_weight)
            if i < size - 1:  # connect to the node South
                graph['connections'].append((current_node, current_node + size))
                graph['weights'].append(fixed_weight)
            if j > 0:  # connect to the node on the West
                graph['connections'].append((current_node, current_node - 1))
                graph['weights'].append(fixed_weight)
            if j < size - 1:  # connect to the node on the East
                graph['connections'].append((current_node, current_node + 1))
                graph['weights'].append(fixed_weight)

    return graph


def generate_bus_line_square_city():
    size=10
    fixed_weight=0.1
    bus_index_multiplier=1000

    distance=fixed_weight * settings["bus_time_travel_cost"]
    cost_get_on = (fixed_weight * settings["wait_for_bus_cost"] *
                                settings["pay_for_bus_cost"] * settings["bus_time_travel_cost"])

    buses_graph=[]
    bus_dict = {
        'name': "bus line UNIQUE",
        'node_map_index': np.array(range(5,96,size)),
        'node_bus_index': np.array(range(5,96,size)) + bus_index_multiplier,
        'connections': [],
        'weights': []
    }

    for row in range(size):
        current_node = (row*10)+5
        bus_current_node = current_node + bus_index_multiplier
        if row < size - 1:  # connect to the node South
                bus_dict['connections'].append((current_node, bus_current_node))
                bus_dict['weights'].append(cost_get_on)
                bus_dict['connections'].append((bus_current_node, bus_current_node + size))
                bus_dict['weights'].append(distance)

    buses_graph.append(bus_dict)
    return buses_graph

def generate_square_city_graph_adjacency_matrix():
    size=10
    distance = 0.1

    num_nodes = size * size
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

    for i in range(size):
        for j in range(size):
            current_node = i * size + j
            if i > 0:  # connect to the node North
                adjacency_matrix[current_node, current_node - size] = distance
            if i < size - 1:  # connect to the node South
                adjacency_matrix[current_node, current_node + size] = distance
            if j > 0:  # connect to the node on the West
                adjacency_matrix[current_node, current_node - 1] = distance
            if j < size - 1:  # connect to the node on the East
                adjacency_matrix[current_node, current_node + 1] = distance

    return adjacency_matrix
