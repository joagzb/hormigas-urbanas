import numpy as np
from ..utils.weights import calculate_bus_time_travel_cost

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
        'node_index': set(range(size * size)),
        'connections': [],
        'weights': []
    }

    for i in range(size):
        for j in range(size):
            current_node = i * size + j

            graph['connections'].append((current_node, list()))
            graph['weights'].append((current_node, list()))

            # Connect to the node North
            if i > 0:
                graph['connections'][-1][1].append(current_node - size)
                graph['weights'][-1][1].append(fixed_weight)

            # Connect to the node South
            if i < size - 1:
                graph['connections'][-1][1].append(current_node + size)
                graph['weights'][-1][1].append(fixed_weight)

            # Connect to the node West
            if j > 0:
                graph['connections'][-1][1].append(current_node - 1)
                graph['weights'][-1][1].append(fixed_weight)

            # Connect to the node East
            if j < size - 1:
                graph['connections'][-1][1].append(current_node + 1)
                graph['weights'][-1][1].append(fixed_weight)

    graph['connections'] = dict(graph['connections'])
    graph['weights'] = dict(graph['weights'])

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

    bus_dict = {
        'name': "bus line UNIQUE",
        'stops': [],  # Connections between map nodes and bus nodes
        'route': np.array(range(5, size*size - size + 5, size)),  # Sequence of nodes in the map the bus will go through
        'node_bus_index': set(np.array(range(5, size*size - size + 5, size)) + bus_node_index_offset),
        'connections': [],
        'weights': []
    }

    for row in range(size - 1):
        current_node = (row * size) + 5
        bus_current_node = current_node + bus_node_index_offset

        # Add bus stop connection
        bus_dict['stops'].append((current_node, bus_current_node))

        # Connect to the node South
        if row < size - 1:
            if (bus_current_node + size) in bus_dict['node_bus_index']:
                bus_dict['connections'].append((bus_current_node, [bus_current_node + size]))
                bus_dict['weights'].append((bus_current_node, [distance]))
            else:
                bus_dict['connections'].append((bus_current_node, []))
                bus_dict['weights'].append((bus_current_node, []))

    bus_dict['connections'] = dict(bus_dict['connections'])
    bus_dict['weights'] = dict(bus_dict['weights'])

    return [bus_dict]
