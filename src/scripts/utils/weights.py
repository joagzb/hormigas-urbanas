import math
from ...configuration.graph_settings import settings

def get_connection_weight(graph, start_node, end_node):
    """
    Retrieves the weight of the connection between two nodes in the graph.

    Parameters:
    graph (dict): The graph dictionary with node indices, connections, and weights.
    start_node (int): The starting node of the connection.
    end_node (int): The ending node of the connection.

    Returns:
    float or None: The weight of the connection if found, None if not found.
    """
    weight_index = graph['connections'][start_node].index(end_node)
    return graph['weights'][start_node][weight_index]

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth's surface.

    Parameters:
    lat1: Latitude of the first point
    lon1: Longitude of the first point
    lat2: Latitude of the second point
    lon2: Longitude of the second point

    Returns:
    distance: Haversine distance between the two points in kilometers
    """
    R = 6371  # Radius of the Earth in kilometers

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_latitude = math.radians(lat2 - lat1)
    delta_longitude = math.radians(lon2 - lon1)

    # Haversine formula
    a = (math.sin(delta_latitude / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_longitude / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def calculate_bus_get_on_cost():
    """Return the cost incurred when boarding a bus.

    The cost combines waiting for the bus and paying the fare. It is
    independent of the route length and bus travel time, ensuring that
    bus usage remains attractive compared to walking.
    """
    return settings["wait_for_bus_cost"] + settings["pay_for_bus_cost"]

def calculate_bus_get_off_cost():
    return settings["bus_get_off"]

def calculate_bus_time_travel_cost(route_weight):
    return route_weight * settings["bus_time_travel_cost"]
