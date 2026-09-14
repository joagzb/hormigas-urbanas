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

## Notebook experiment

Open `src/TPF.ipynb` from either the repository root or `src/`. The notebook locates the repository root before importing project modules or creating outputs. It takes `ants`, `epomax`, ACO global-best patience, and the ACS/BWAS path-consensus threshold from `src/configuration/algorithm_settings.py`; its default experiment keeps the 10x10 graph with 20 ants, 100 maximum epochs, 10-epoch ACO patience, and an 85% ACS/BWAS threshold.

Plotly figures are not displayed inline. Each algorithm records bounded JSONL history and writes one interactive animation to the repository `tmp/` directory: `aco_pheromone_animation.html`, `acs_pheromone_animation.html`, and `bwas_pheromone_animation.html`. The notebook prints each generated HTML path plus separate algorithm/history and HTML build/write timings.

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
figure = draw_pheromone_history(graph, history_path, reference_path=reference_path, stride=5, max_frames=60, show=False)
figure.write_html('tmp/pheromone_animation.html')
```

`epoch_callback` is an optional keyword-only algorithm argument and receives one observation dictionary per completed epoch. It observes the final `pheromone_update` state only after every pheromone mutation for that epoch has finished. For BWAS this includes worst-path penalties, mutation, floor enforcement, and any restart, so the saved state is the graph used by the next epoch.

Simple ACO stops after 10 consecutive completed epochs without a strict finite global-best cost improvement, configurable with `aco_global_best_patience`; the first finite best and every later strict improvement reset the counter. It cannot stop before epoch 2. When derived pheromone initialization (`initial_pheromone_lvl=None`) finds no finite deterministic baseline route, ACO returns safely with no route at epoch 0 because it cannot derive a meaningful initial pheromone level; otherwise `max_epochs` remains its hard cap. ACS and BWAS may stop after epoch 2 when at least `ceil(path_consensus_threshold * finite_ant_count)` finite ants completed the same exact route and the finite iteration-best cost is exactly unchanged from the preceding epoch. Lost ants are excluded from the denominator, equal-cost alternative routes are distinct, and `max_epochs` remains every algorithm's hard cap. Dijkstra is an external reference, not a termination target. BWAS can still restart pheromones after configured non-improving epochs to restore diversity while preserving its global best; the restart does not terminate the search.

`PheromoneHistoryWriter` truncates its target when created, writes a versioned header, then appends one strict JSON record per epoch. Edges use stable `(source, adjacency_index)` order and routes use graph-relative node indices, so directed/parallel edges and opaque mixed IDs remain aligned without pickle or `eval`. Loading validates the graph fingerprint and streams the file while retaining the first and final frames within `max_frames`; use the same graph instance or an exactly equivalent graph for recording and drawing. The animation shows a static structural-edge backdrop followed by the Dijkstra reference route, global-best-found route, and nodes. Edges expose no hover metadata or legend entry, and iteration appears only in the title and slider. Plotly HTML exports do not require Kaleido.

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
