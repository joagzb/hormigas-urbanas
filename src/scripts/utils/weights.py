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
    for conn, weight in zip(graph['connections'], graph['weights']):
        if conn == (start_node, end_node):
            return weight
    return None