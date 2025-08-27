# Urban Ants

An optimal route-finding algorithm between two points in a city, minimizing both time and cost while considering urban public buses. The algorithm has been built following Ant Colony Optimization (ACO) methodology.

More details about the methodology can be found at:
- [Paper work](./docs/hormigas%20urbanas%20-%202018.pdf)
- [Joaquin's portfolio](https://joagzb.com)

## Technologies and libraries

- Python 3.x
- NumPy (core algorithms)
- OpenStreetMap data
- NetworkX
- Matplotlib

## Getting started

1. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
2. **Install dependencies**
   ```bash
   pip install -r src/requirements.txt
   ```
3. **Run a simple check**
   The repository includes small utilities that can be executed as Python modules. For example:
   ```bash
   python -m src.scripts.test.test_graph_generators
   ```
   The script generates a toy square city, adds a bus line and prints an example
   route demonstrating that, thanks to a fixed boarding cost (waiting + paying),
   taking the bus is noticeably cheaper than walking the same distance.

## Graph visualization

Generate a random graph, calculate a route, and visualize it:

```python
from src.scripts.utils.generators import generate_random_graph
from src.scripts.utils.route_finder import dijkstra
from src.scripts.utils.graph_visualizer import draw_graph

graph = generate_random_graph(10)
path = dijkstra(graph, 0, 5)
draw_graph(graph, path)
```

You can also run the module directly:

```bash
python -m src.scripts.utils.graph_visualizer
```

Bus nodes (IDs ≥ 1000) are shown in orange, and the selected path is drawn in red.

## Running tests

After installing the dependencies, you can run the test suite with [pytest](https://docs.pytest.org/en/stable/):

```bash
python -m pip install pytest
pytest
```

## Authors

- Joaquin Gonzalez Budiño: <joa_gzb@hotmail.com>
- Nicolas Giuliano: <nsgiuliano@gmail.com>

## Social networking

- [LinkedIn Joaquin](https://www.linkedin.com/in/joaquin-gonzalez-budino/)
- [Joaquin's portfolio](https://joagzb.com)
- [LinkedIn Nicolas](https://www.linkedin.com/in/nicolás-giuliano-204a301a4/)

## License

- [GPLv3](./LICENSE)
