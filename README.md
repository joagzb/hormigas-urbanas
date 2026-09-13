# Urban Ants

A route-finding experiment between two points in a city using walking and urban buses. Edge weights form one static generalized-cost score; they are not calibrated travel minutes or converted fares, and the model does not represent schedules or headways.

More details about the methodology can be found at:
- [Joaquin's portfolio](https://joagzb.com)

## Technologies and libraries

- Python 3.x
- NumPy (core algorithms)
- NetworkX
- Plotly

## Getting started

1. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `source .venv/Scripts/activate` on a cmd or git bash terminal
   ```
2. **Install dependencies**
   ```bash
   pip install -r src/requirements.txt
   ```
3. **Run the interactive toy-city demo from the repository root**
   ```bash
   python -m src.scripts.utils.toy_city_generators
   ```
   The equivalent command from inside `src/` is `python -m scripts.utils.toy_city_generators`.
   The script generates a 10x10 city, prompts for start/end nodes, and writes two interactive HTML route files.

## Notebook experiment profiles

Open `src/TPF.ipynb` with `src/` as the working directory. The installed Plotly, `nbformat`, and IPython dependencies allow the notebook's `figure.show()` calls to render inline.

The notebook defaults to its clearly labeled **fast demo** profile (10x10, 10 ants, 10 epochs) for interactive use. Set `RUN_FULL_EXPERIMENT = True` to deliberately run the original **full experiment** (20x20, 50 ants, 500 epochs). The full experiment is long-running: route construction explores the much larger stochastic graph once per ant and epoch, so it can take several minutes or longer. JSONL history remains bounded on disk and animation rendering is timed separately from algorithm execution.

## Graph visualization

Given a validated `graph` dictionary, valid `start_node` and `end_node` IDs, and ACO parameters chosen for the experiment, this integration snippet writes post-update epochs to an inspectable JSON Lines file and visualizes them:

```python
from src.scripts.ant_colony_simple_ACO.ant_colony_optimization import ACO
from src.scripts.utils.graph_visualizer import PheromoneHistoryWriter, draw_pheromone_history
from src.scripts.utils.route_finder import dijkstra

reference_path = dijkstra(graph, start_node, end_node)
history_path = 'tmp/aco_pheromone_history.jsonl'
history_writer = PheromoneHistoryWriter(graph, history_path)
best_path, best_cost, elapsed, epochs = ACO(graph, start_node, end_node, 20, 0.1, None, 1.0, 2.0, epoch_callback=history_writer)
figure = draw_pheromone_history(graph, history_path, reference_path=reference_path, stride=5, max_frames=60)
# figure.write_html('pheromone_animation.html')
```

`epoch_callback` is an optional keyword-only algorithm argument and receives one observation dictionary per completed epoch. It observes the final `pheromone_update` state only after every pheromone mutation for that epoch has finished. For BWAS this includes worst-path penalties, mutation, floor enforcement, and any restart, so the saved state is the graph used by the next epoch.

`PheromoneHistoryWriter` truncates its target when created, writes a versioned header, then appends one strict JSON record per epoch. Edges use stable `(source, adjacency_index)` order and routes use graph-relative node indices, so directed/parallel edges and opaque mixed IDs remain aligned without pickle or `eval`. Loading validates the graph fingerprint and streams the file while retaining the first and final frames within `max_frames`; use the same graph instance or an exactly equivalent graph for recording and drawing. Bus edge metadata controls styling, and the optional Dijkstra overlay is the shortest-route reference under the configured generalized costs. Plotly HTML and notebook rendering do not require Kaleido.

## Running tests

After installing the dependencies, 

```bash
   pip install pytest  # required for running the tests
   ```

you can run the test suite with [pytest](https://docs.pytest.org/en/stable/) by directly running:

```bash
pytest
```

## Authors

- Joaquin Gonzalez Budiño: <joa_gzb@hotmail.com>
- Nicolas Giuliano: <nsgiuliano@gmail.com>

- [LinkedIn Joaquin](https://www.linkedin.com/in/joaquin-gonzalez-budino/)
- [LinkedIn Nicolas](https://www.linkedin.com/in/nicolás-giuliano-204a301a4/)

## License

- [GPLv3](./LICENSE)
