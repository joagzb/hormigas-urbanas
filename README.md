# Urban Ants

A route-finding experiment between two points in a city using walking and urban buses. Edge weights form one static generalized-cost score; they are not calibrated travel minutes or converted fares, and the model does not represent schedules or headways.

More details about the methodology can be found at:
- [Joaquin's portfolio](https://joagzb.com)

## Technologies and libraries

- Python 3.x
- NumPy (core algorithms)
- NetworkX
- Matplotlib

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
3. **Run a simple check**
   The repository includes a quick example that can be executed as Python modules:
   ```bash
   cd src
   python -m scripts.utils.toy_city_generators
   ```
   The script generates a toy square city (10x10), adds outbound and inbound bus services, and prints an example
   route demonstrating that.

## Graph visualization

Generate a random graph, calculate a route, and visualize it:

```python
from src.scripts.utils.route_finder import dijkstra
from src.scripts.utils.graph_visualizer import draw_graph

path = dijkstra(graph, 0, 5)
draw_graph(graph, path)
```

Bus nodes use deterministic opaque IDs such as `bus:UNIQUE:outbound:0`. Service metadata identifies them for orange styling, and the selected path is drawn in red. Graph edges carry aligned `walk`, `board`, `ride`, or `alight` metadata.

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
