import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from .generators import validate_graph


@dataclass(frozen=True)
class GraphEdge:
  source: Hashable
  adjacency_index: int
  target: Hashable
  edge_type: str
  weight: float


def _node_sort_key(node):
  return type(node).__module__, type(node).__qualname__, repr(node)


def _requires_edge_types(graph_dict):
  return bool(graph_dict.get('buses')) or any(isinstance(node, str) and node.startswith('bus:') for node in graph_dict.get('node_index', []))


def stable_edge_order(graph_dict: dict) -> Tuple[GraphEdge, ...]:
  """Return every directed edge without collapsing parallel adjacency entries."""
  validate_graph(graph_dict, require_edge_types=_requires_edge_types(graph_dict))
  edge_types = graph_dict.get('edge_types')
  edges = []
  for source in sorted(graph_dict['connections'], key=_node_sort_key):
    neighbors = graph_dict['connections'][source]
    weights = graph_dict['weights'][source]
    source_edge_types = edge_types[source] if edge_types is not None else ['walk'] * len(neighbors)
    for adjacency_index, (target, weight, edge_type) in enumerate(zip(neighbors, weights, source_edge_types)):
      edges.append(GraphEdge(source, adjacency_index, target, edge_type, weight))
  return tuple(edges)


HISTORY_FORMAT = 'urban-ants-pheromone-history'
HISTORY_VERSION = 1


def calculate_animation_stride(total_iterations):
  """Return a bounded stride that keeps approximately 100 animation frames."""
  return max(1, min(10, math.ceil(total_iterations / 100)))


def _node_order(graph_dict):
  return tuple(sorted(graph_dict['node_index'], key=_node_sort_key))


def _json_number(value):
  value = float(value)
  if math.isnan(value):
    raise ValueError('Pheromone history cannot encode NaN values')
  return value if math.isfinite(value) else None


def _finite_pheromone_number(value):
  try:
    value = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError('Pheromone history pheromones must contain finite numeric values') from error
  if not math.isfinite(value):
    raise ValueError('Pheromone history pheromones must contain finite numeric values')
  return value


def _graph_history_header(graph_dict, edges, nodes):
  node_positions = {node: index for index, node in enumerate(nodes)}
  fingerprint_payload = {
    'nodes': [[type(node).__module__, type(node).__qualname__, repr(node)] for node in nodes],
    'edges': [[node_positions[edge.source], edge.adjacency_index, node_positions[edge.target], _json_number(edge.weight), edge.edge_type] for edge in edges],
  }
  encoded_payload = json.dumps(fingerprint_payload, allow_nan=False, separators=(',', ':')).encode('utf-8')
  return {
    'type': HISTORY_FORMAT,
    'version': HISTORY_VERSION,
    'node_count': len(nodes),
    'edge_count': len(edges),
    'row_lengths': [len(graph_dict['connections'][node]) for node in nodes],
    'fingerprint': hashlib.sha256(encoded_payload).hexdigest(),
  }


class PheromoneHistoryWriter:
  """Write aligned final post-update epochs to a caller-selected JSONL path.

  The graph may contain opaque mixed node IDs and parallel directed edges;
  records encode both through stable graph-relative indices. Construction
  creates the target parent directory and truncates the file. Calls append one
  complete epoch and reject misaligned pheromones or paths outside the graph.
  """

  def __init__(self, graph_dict: dict, history_path):
    self.edge_order = stable_edge_order(graph_dict)
    self.node_order = _node_order(graph_dict)
    self.node_positions = {node: index for index, node in enumerate(self.node_order)}
    self.history_path = Path(history_path)
    self.history_path.parent.mkdir(parents=True, exist_ok=True)
    self._last_epoch = 0
    header = _graph_history_header(graph_dict, self.edge_order, self.node_order)
    self.history_path.write_text(json.dumps(header, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')

  def __call__(self, observation):
    stage = observation.get('stage', 'pheromone_update')
    if stage != 'pheromone_update':
      raise ValueError('Pheromone history only accepts final pheromone_update observations')
    epoch = int(observation['epoch'])
    if epoch == self._last_epoch:
      return
    if epoch <= self._last_epoch:
      raise ValueError('Pheromone history epochs must be strictly increasing')

    pheromone_graph = observation['pheromones']
    pheromones = []
    for edge in self.edge_order:
      try:
        value = pheromone_graph[edge.source][edge.adjacency_index]
      except (KeyError, IndexError, TypeError) as error:
        raise ValueError('Pheromones must remain aligned with graph connections') from error
      pheromones.append(_finite_pheromone_number(value))

    record = {
      'epoch': epoch,
      'stage': stage,
      'pheromones': pheromones,
      'iteration_best_path': self._encode_path(observation.get('iteration_best_path')),
      'iteration_best_cost': _json_number(observation.get('iteration_best_cost', np.inf)),
      'global_best_path': self._encode_path(observation.get('global_best_path')),
      'global_best_cost': _json_number(observation.get('global_best_cost', np.inf)),
      'restarted': bool(observation.get('restarted', False)),
    }
    with self.history_path.open('a', encoding='utf-8') as history_file:
      history_file.write(json.dumps(record, allow_nan=False, separators=(',', ':')) + '\n')
    self._last_epoch = epoch

  def _encode_path(self, path):
    if path is None:
      return None
    try:
      return [self.node_positions[node] for node in path]
    except (KeyError, TypeError) as error:
      raise ValueError('Observed paths must contain graph nodes only') from error


class ExperimentOutputWriter:
  """Own one algorithm's generated history and animation under repository ``tmp``.

  ``graph_dict`` supplies stable edge and opaque-node alignment, while ``name``
  becomes the generated file prefix. Construction creates repository ``tmp``
  and a private ``PheromoneHistoryWriter``. Calling the instance records an
  epoch; ``write_animation`` loads that history and writes the matching HTML.
  """

  def __init__(self, graph_dict: dict, name: str):
    if not name or any(character not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for character in name):
      raise ValueError('output name must contain lowercase letters, numbers, underscores, or hyphens only')
    self.graph_dict = graph_dict
    self.name = name
    self.output_directory = Path(__file__).resolve().parents[3] / 'tmp'
    self.output_directory.mkdir(parents=True, exist_ok=True)
    self.history_path = self.output_directory / f'{name}_pheromone_history.jsonl'
    self.html_path = self.output_directory / f'{name}_pheromone_animation.html'
    self._history_writer = PheromoneHistoryWriter(graph_dict, self.history_path)

  def __call__(self, observation):
    """Append one final algorithm observation to this output's JSONL history."""
    self._history_writer(observation)

  def write_animation(self, *, reference_path=None, title='Pheromone evolution', route_label='Global best-found route', stride=None, max_frames=100, pheromone_bins=6):
    """Build and write the configured HTML animation, returning its path.

    ``reference_path`` may contain the graph's opaque IDs. Sampling arguments
    are forwarded to the validated history loader. This method performs file
    I/O but never displays the Plotly figure inline.
    """
    resolved_stride = calculate_animation_stride(self._history_writer._last_epoch) if stride is None else stride
    figure = draw_pheromone_history(
      self.graph_dict,
      self.history_path,
      reference_path=reference_path,
      title=title,
      route_label=route_label,
      stride=resolved_stride,
      max_frames=max_frames,
      pheromone_bins=pheromone_bins,
      show=False,
    )
    figure.write_html(self.html_path)
    return self.html_path


def _read_json_line(line, line_number):
  try:
    value = json.loads(line)
  except json.JSONDecodeError as error:
    raise ValueError(f'Invalid pheromone history JSON on line {line_number}') from error
  if not isinstance(value, dict):
    raise ValueError(f'Pheromone history line {line_number} must be an object')
  return value


def _validate_history_header(header, expected_header):
  for field in ('type', 'version', 'node_count', 'edge_count', 'row_lengths', 'fingerprint'):
    if header.get(field) != expected_header[field]:
      raise ValueError(f'Pheromone history graph alignment mismatch: {field}')


def _record_number(value, field):
  if value is None:
    return np.inf
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError(f'Pheromone history {field} must be numeric or null')
  value = float(value)
  if not math.isfinite(value):
    raise ValueError(f'Pheromone history {field} must use null for infinities')
  return value


def _record_pheromone_number(value):
  if isinstance(value, bool) or not isinstance(value, (int, float)):
    raise ValueError('Pheromone history pheromones must contain finite numeric values')
  return _finite_pheromone_number(value)


def _decode_path(path, nodes, field):
  if path is None:
    return None
  if not isinstance(path, list):
    raise ValueError(f'Pheromone history {field} must be a list or null')
  decoded = []
  for node_index in path:
    if isinstance(node_index, bool) or not isinstance(node_index, int) or not 0 <= node_index < len(nodes):
      raise ValueError(f'Pheromone history {field} contains an invalid node index')
    decoded.append(nodes[node_index])
  return decoded


def _decode_history_record(record, edges, nodes, previous_epoch):
  epoch = record.get('epoch')
  if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= previous_epoch:
    raise ValueError('Pheromone history epochs must be strictly increasing integers')
  if record.get('stage') != 'pheromone_update':
    raise ValueError('Pheromone history records must use stage pheromone_update')
  values = record.get('pheromones')
  if not isinstance(values, list) or len(values) != len(edges):
    raise ValueError('Pheromone history pheromones must match the stable edge order')
  restarted = record.get('restarted', False)
  if not isinstance(restarted, bool):
    raise ValueError('Pheromone history restarted must be boolean')
  return {
    'epoch': epoch,
    'stage': 'pheromone_update',
    'pheromones': np.asarray([_record_pheromone_number(value) for value in values], dtype=float),
    'iteration_best_path': _decode_path(record.get('iteration_best_path'), nodes, 'iteration_best_path'),
    'iteration_best_cost': _record_number(record.get('iteration_best_cost'), 'iteration_best_cost'),
    'global_best_path': _decode_path(record.get('global_best_path'), nodes, 'global_best_path'),
    'global_best_cost': _record_number(record.get('global_best_cost'), 'global_best_cost'),
    'restarted': restarted,
  }


def load_pheromone_history(graph_dict: dict, history_path, *, stride: int = 1, max_frames: int = 100):
  """Stream and validate JSONL history, retaining sampled first/final frames."""
  if stride < 1:
    raise ValueError('stride must be at least 1')
  if max_frames < 2:
    raise ValueError('max_frames must be at least 2 to retain first and final epochs')
  edges = stable_edge_order(graph_dict)
  nodes = _node_order(graph_dict)
  expected_header = _graph_history_header(graph_dict, edges, nodes)
  sampled = []
  latest = None
  previous_epoch = 0

  with Path(history_path).open(encoding='utf-8') as history_file:
    header_line = history_file.readline()
    if not header_line:
      raise ValueError('Pheromone history is empty')
    _validate_history_header(_read_json_line(header_line, 1), expected_header)
    for line_number, line in enumerate(history_file, start=2):
      if not line.strip():
        raise ValueError(f'Pheromone history line {line_number} is empty')
      snapshot = _decode_history_record(_read_json_line(line, line_number), edges, nodes, previous_epoch)
      previous_epoch = snapshot['epoch']
      latest = snapshot
      if not sampled:
        sampled.append(snapshot)
      elif snapshot['epoch'] % stride == 0 and len(sampled) < max_frames - 1:
        sampled.append(snapshot)

  if latest is not None and latest['epoch'] != sampled[-1]['epoch']:
    sampled.append(latest)
  return sampled


def _copy_path(path):
  return None if path is None else list(path)


def build_graph_from_dict(graph_dict: dict) -> nx.DiGraph:
  """Create a NetworkX graph for metadata and deterministic node layout."""
  validate_graph(graph_dict, require_edge_types=_requires_edge_types(graph_dict))
  graph_nx = nx.DiGraph()

  for node in graph_dict['node_index']:
    graph_nx.add_node(node)

  for bus in graph_dict.get('buses', []):
    for map_node, bus_node in bus['stops']:
      graph_nx.nodes[bus_node]['node_type'] = 'bus'
      graph_nx.nodes[bus_node]['map_node'] = map_node
      graph_nx.nodes[bus_node]['line_id'] = bus['line_id']
      graph_nx.nodes[bus_node]['direction'] = bus['direction']

  for edge in stable_edge_order(graph_dict):
    graph_nx.add_edge(edge.source, edge.target, weight=edge.weight, edge_type=edge.edge_type)
  return graph_nx


def _grid_positions(graph_dict: dict, graph_nx: nx.DiGraph) -> Optional[Dict[Hashable, Tuple[float, float]]]:
  """Return a grid layout when map nodes are 0..n*n-1; otherwise None."""
  try:
    node_index_obj = graph_dict.get('node_index')
    if isinstance(node_index_obj, (set, list)):
      nodes = list(node_index_obj)
    else:
      nodes = list(graph_nx.nodes)

    base_nodes: List[int] = []
    for node in nodes:
      if graph_nx.nodes[node].get('node_type') != 'bus' and isinstance(node, int):
        base_nodes.append(node)
    if not base_nodes:
      return None

    max_base = max(base_nodes)
    if set(base_nodes) != set(range(max_base + 1)):
      return None

    side = math.isqrt(max_base + 1)
    if side * side != max_base + 1:
      return None

    positions: Dict[Hashable, Tuple[float, float]] = {}
    for node in base_nodes:
      row, column = divmod(node, side)
      positions[node] = (column, -row)

    bus_nodes = [node for node in nodes if graph_nx.nodes[node].get('node_type') == 'bus']
    for index, node in enumerate(sorted(bus_nodes, key=_node_sort_key)):
      base = graph_nx.nodes[node]['map_node']
      if base in positions:
        x, y = positions[base]
        direction_offset = 0.12 if graph_nx.nodes[node]['direction'] == 'outbound' else -0.12
        positions[node] = (x + 0.25 + (index % 3) * 0.04, y + direction_offset)
      else:
        positions[node] = (0.0, 0.0)
    return positions
  except (KeyError, TypeError, ValueError):
    return None


def compute_positions(graph_dict: dict, graph_nx: nx.DiGraph) -> Dict[Hashable, Tuple[float, float]]:
  """Compute node positions with grid preference and spring fallback."""
  return _grid_positions(graph_dict, graph_nx) or nx.spring_layout(graph_nx, seed=42)


def node_style(graph_nx: nx.DiGraph, base_size: int = 300, bus_size: int = 450) -> Tuple[List[str], List[int]]:
  """Return compatibility color and size sequences for graph nodes."""
  colors = ['orange' if graph_nx.nodes[node].get('node_type') == 'bus' else 'lightblue' for node in graph_nx.nodes]
  sizes = [bus_size if graph_nx.nodes[node].get('node_type') == 'bus' else base_size for node in graph_nx.nodes]
  return colors, sizes


def _route_trace(path, positions, *, name, color, dash='solid', width=4):
  x_values = []
  y_values = []
  marker_sizes = []
  if path:
    for source, target in zip(path, path[1:]):
      if source not in positions or target not in positions:
        continue
      source_x, source_y = positions[source]
      target_x, target_y = positions[target]
      x_values.extend((source_x, target_x, None))
      y_values.extend((source_y, target_y, None))
      marker_sizes.extend((0, 9, 0))
  return go.Scatter(
    x=x_values,
    y=y_values,
    mode='lines+markers',
    line={'color': color, 'dash': dash, 'width': width},
    marker={'color': color, 'size': marker_sizes, 'symbol': 'arrow', 'angleref': 'previous'},
    hoverinfo='skip',
    name=name,
  )


def _structural_edge_trace(edges, positions):
  x_values = []
  y_values = []
  for edge in edges:
    source_x, source_y = positions[edge.source]
    target_x, target_y = positions[edge.target]
    x_values.extend((source_x, target_x, None))
    y_values.extend((source_y, target_y, None))
  return go.Scatter(x=x_values, y=y_values, mode='lines', line={'color': '#7f8c8d', 'width': 1}, opacity=0.35, hoverinfo='skip', name='Structural graph edges', showlegend=False)


def _node_trace(graph_nx, positions):
  nodes = list(graph_nx.nodes)
  colors = ['#f39c12' if graph_nx.nodes[node].get('node_type') == 'bus' else '#aed6f1' for node in nodes]
  sizes = [13 if graph_nx.nodes[node].get('node_type') == 'bus' else 10 for node in nodes]
  return go.Scatter(
    x=[positions[node][0] for node in nodes],
    y=[positions[node][1] for node in nodes],
    mode='markers+text' if len(nodes) <= 400 else 'markers',
    marker={'color': colors, 'size': sizes, 'line': {'color': '#34495e', 'width': 0.5}},
    text=[str(node) for node in nodes],
    textposition='top center',
    hoverinfo='skip',
    name='Nodes',
  )


def build_pheromone_animation(graph_dict: dict, snapshots, *, reference_path=None, title='Pheromone evolution', route_label='Global best-found route', pheromone_bins=6):
  """Build a minimal route animation from bounded post-update snapshots.

  ``reference_path`` is supplied externally and represents a Dijkstra shortest
  route under the graph's configured generalized costs.
  """
  if pheromone_bins < 1:
    raise ValueError('pheromone_bins must be at least 1')
  edges = stable_edge_order(graph_dict)
  snapshots = list(snapshots)

  graph_nx = build_graph_from_dict(graph_dict)
  positions = compute_positions(graph_dict, graph_nx)

  if snapshots:
    for snapshot in snapshots:
      if len(snapshot['pheromones']) != len(edges):
        raise ValueError('Snapshot pheromones must match the stable edge order')
      pheromone_array = np.asarray(snapshot['pheromones'], dtype=float)
      if not np.all(np.isfinite(pheromone_array)):
        raise ValueError('Snapshot pheromones must contain finite numeric values')
  else:
    snapshots = [
      {
        'epoch': 0,
        'stage': 'initial',
        'pheromones': np.zeros(len(edges)),
        'iteration_best_path': None,
        'iteration_best_cost': np.inf,
        'global_best_path': None,
        'global_best_cost': np.inf,
        'restarted': False,
      }
    ]

  def frame_traces(snapshot):
    return [
      _structural_edge_trace(edges, positions),
      _route_trace(reference_path, positions, name='Dijkstra reference route', color='#2ecc71', dash='dash', width=3),
      _route_trace(snapshot.get('global_best_path'), positions, name=route_label, color='#e74c3c'),
      _node_trace(graph_nx, positions),
    ]

  has_recorded_epochs = snapshots[0]['epoch'] != 0
  frames = []
  slider_steps = []
  if has_recorded_epochs:
    for snapshot in snapshots:
      frame_name = str(snapshot['epoch'])
      frame_label = f'Iteration {snapshot["epoch"]}'
      frame_title = f'{title} - {frame_label}'
      frames.append(go.Frame(name=frame_name, data=frame_traces(snapshot), layout=go.Layout(title={'text': frame_title})))
      slider_steps.append(
        {'label': frame_label, 'method': 'animate', 'args': [[frame_name], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}]}
      )

  first_snapshot = snapshots[0]
  layout = go.Layout(
    title={'text': f'{title} - Iteration {first_snapshot["epoch"]}'},
    hovermode=False,
    template='plotly_white',
    showlegend=True,
    xaxis={'visible': False, 'scaleanchor': 'y', 'scaleratio': 1},
    yaxis={'visible': False},
    margin={'l': 20, 'r': 20, 't': 70, 'b': 90},
  )
  if has_recorded_epochs:
    layout.updatemenus = [
      {
        'type': 'buttons',
        'direction': 'left',
        'showactive': False,
        'x': 0,
        'y': -0.08,
        'buttons': [
          {'label': 'Play', 'method': 'animate', 'args': [None, {'fromcurrent': True, 'frame': {'duration': 350, 'redraw': False}, 'transition': {'duration': 100}}]},
          {'label': 'Pause', 'method': 'animate', 'args': [[None], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': False}, 'transition': {'duration': 0}}]},
        ],
      }
    ]
    layout.sliders = [{'active': 0, 'currentvalue': {'prefix': ''}, 'pad': {'t': 45}, 'steps': slider_steps}]

  return go.Figure(data=frame_traces(first_snapshot), layout=layout, frames=frames)


def draw_pheromone_history(
  graph_dict: dict,
  history_path,
  *,
  reference_path=None,
  title='Pheromone evolution',
  route_label='Global best-found route',
  stride: int = 1,
  max_frames: int = 100,
  pheromone_bins: int = 6,
  show: bool = True,
):
  """Load a validated JSONL history and build its Plotly animation."""
  snapshots = load_pheromone_history(graph_dict, history_path, stride=stride, max_frames=max_frames)
  figure = build_pheromone_animation(graph_dict, snapshots, reference_path=reference_path, title=title, route_label=route_label, pheromone_bins=pheromone_bins)
  if show:
    figure.show()
  return figure


def draw_graph(
  graph: dict,
  path: Optional[List[Hashable]] = None,
  save_path: Optional[str] = None,
  *,
  reference_path: Optional[List[Hashable]] = None,
  title: str = 'Directed city graph',
  route_label: str = 'Global best-found route',
):
  """Display a Plotly graph and optionally highlight a best-found route.

  Interactive exports use HTML and do not require Kaleido.
  """
  snapshot = {
    'epoch': 0,
    'stage': 'initial',
    'pheromones': np.ones(len(stable_edge_order(graph))),
    'iteration_best_path': None,
    'global_best_path': _copy_path(path),
    'restarted': False,
  }
  figure = build_pheromone_animation(graph, [snapshot], reference_path=reference_path, title=title, route_label=route_label)
  if save_path:
    output_path = Path(save_path)
    if output_path.suffix.lower() not in {'.html', '.htm'}:
      raise ValueError('Plotly graph exports must use an .html or .htm path')
    figure.write_html(output_path)
  else:
    figure.show()
  return figure
