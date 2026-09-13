import heapq
from itertools import count

from .generators import validate_graph


def dijkstra(graph, start_node, end_node):
  """
  Finds one of the best routes between two nodes using Dijkstra's algorithm.

  Parameters:
  graph (dict): The graph dictionary with node indices, connections, and weights.
  start_node: The starting node.
  end_node: The ending node.

  Returns:
  list: The route from start_node to end_node.
  """
  validate_graph(graph, require_edge_types=False)
  if start_node not in graph['node_index'] or end_node not in graph['node_index']:
    raise ValueError('Start and end nodes must exist in the graph')

  # The sequence prevents heap ties from comparing opaque mixed node IDs.
  sequence = count()
  priority_queue = [(0, next(sequence), start_node, [])]
  visited = set()
  distances = {node: float('inf') for node in graph['node_index']}
  distances[start_node] = 0

  while priority_queue:
    (current_distance, _, current_node, path) = heapq.heappop(priority_queue)

    # If the end node is reached, return the path
    if current_node == end_node:
      return path + [end_node]

    # Skip if the node has been visited
    if current_node in visited:
      continue

    visited.add(current_node)
    path = path + [current_node]

    # Check all neighbors of the current node
    if current_node in graph['connections']:
      neighbors = graph['connections'][current_node]
      weights = graph['weights'][current_node]
      for i, neighbor in enumerate(neighbors):
        if neighbor in graph['node_index']:  # Ensure neighbor is within node_index
          weight = weights[i]
          distance = current_distance + weight
          if distance < distances[neighbor]:
            distances[neighbor] = distance
            heapq.heappush(priority_queue, (distance, next(sequence), neighbor, path))

  return None  # Return None if no path is found
