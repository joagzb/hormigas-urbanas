import heapq

def dijkstra(graph, start_node, end_node):
    """
    Finds one of the best routes between two nodes using Dijkstra's algorithm.

    Parameters:
    graph (dict): The graph dictionary with node indices, connections, and weights.
    start_node (int): The starting node.
    end_node (int): The ending node.

    Returns:
    list: The route from start_node to end_node.
    """
    # Initialize the priority queue
    priority_queue = [(0, start_node, [])]
    visited = set()
    distances = {node: float('inf') for node in graph['node_index']}
    distances[start_node] = 0

    while priority_queue:
        (current_distance, current_node, path) = heapq.heappop(priority_queue)

        # If the end node is reached, return the path
        if current_node == end_node:
            return path + [end_node]

        # Skip if the node has been visited
        if current_node in visited:
            continue

        visited.add(current_node)
        path = path + [current_node]

        # Check all neighbors of the current node
        neighbors = [(conn[1], weight) for conn, weight in zip(graph['connections'], graph['weights']) if conn[0] == current_node]
        for neighbor, weight in neighbors:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor, path))

    return None  # Return None if no path is found