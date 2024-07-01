import math

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