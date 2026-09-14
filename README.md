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

Open `src/TPF.ipynb` from either the repository root or `src/` after activating the project environment. The notebook does not rewrite import or output roots. It takes `ants`, `epomax`, and shared global-best patience from `src/configuration/algorithm_settings.py`; the default experiment uses a 10x10 graph, 20 ants, 100 maximum epochs, and 10-epoch patience.

Plotly figures are not displayed inline. `ExperimentOutputWriter` creates `<working_directory>/tmp`, records each algorithm's JSONL history, and writes its interactive animation there. The notebook prints each generated HTML path plus separate algorithm/history and HTML build/write timings.

## Graph visualization

Given a validated `graph` dictionary, valid `start_node` and `end_node` IDs, and ACO parameters chosen for the experiment, this integration snippet writes post-update epochs to an inspectable JSON Lines file and visualizes them:

```python
from src.scripts.ant_colony_simple_ACO.ant_colony_optimization import ACO
from src.scripts.main import prepare_routing_problem
from src.scripts.utils.graph_visualizer import ExperimentOutputWriter
from src.scripts.utils.dijkstra import dijkstra

problem = prepare_routing_problem(graph, start_node, end_node)
reference_path = dijkstra(problem.graph, start_node, end_node)
output = ExperimentOutputWriter(problem.graph, 'aco')
best_path, best_cost, elapsed, epochs = problem.run(ACO, 20, 0.1, None, 1.0, 2.0, global_best_patience=10, epoch_callback=output)
html_path = output.write_animation(reference_path=reference_path)
```

`epoch_callback` is an optional keyword-only algorithm argument and receives one observation dictionary per completed epoch. It observes the final `pheromone_update` state only after every pheromone mutation for that epoch has finished. For BWAS this includes worst-path penalties, mutation, floor enforcement, and any restart, so the saved state is the graph used by the next epoch.

ACO, ACS, and BWAS stop after 10 consecutive completed epochs without a strict global-best cost improvement, configurable with `global_best_patience`, or at `max_epochs`. The first finite best and every later strict improvement reset the counter. When automatic pheromone initialization cannot derive a finite deterministic baseline route, an algorithm returns safely at epoch 0. The baseline is used only for each variant's tau0 formula; Dijkstra remains an external reference and ant costs remain sums of original aligned edge weights. BWAS can restart pheromones after its separately configured non-improving interval to restore diversity while retaining its global best; a restart is not termination.

`ExperimentOutputWriter` owns the generated paths and delegates strict JSONL recording to `PheromoneHistoryWriter`. Edges use stable `(source, adjacency_index)` order and routes use graph-relative node indices, so directed/parallel edges and opaque mixed IDs remain aligned without pickle or `eval`. Loading validates the graph fingerprint and streams the file while retaining the first and final frames within `max_frames`; use the same graph instance or an exactly equivalent graph for recording and drawing. The default animation stride adapts to the completed run length: runs through 100 iterations show every frame, while longer runs progressively sample up to a maximum stride of 10. The animation shows a static structural-edge backdrop followed by the Dijkstra reference route, global-best-found route, and nodes. Edges expose no hover metadata or legend entry, and iteration appears only in the title and slider. Plotly HTML exports do not require Kaleido.

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
