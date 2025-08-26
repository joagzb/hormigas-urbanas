# Urban Ants

An optimal route-finding algorithm between two points in a city, minimizing both time and cost while considering urban public buses. The algorithm has been built following Ant Colony Optimization (ACO) methodology.

More details about the methodology can be found at:
- [Paper work](./docs/hormigas%20urbanas%20-%202018.pdf)
- [Joaquin's portfolio](https://joagzb.com)

## Technologies and libraries

- Python 3.x
- NumPy (core algorithms)
- OpenStreetMap data

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

## Authors

- Joaquin Gonzalez Budiño: <joa_gzb@hotmail.com>
- Nicolas Giuliano: <nsgiuliano@gmail.com>

## Social networking

- [LinkedIn Joaquin](https://www.linkedin.com/in/joaquin-gonzalez-budino/)
- [Joaquin's portfolio](https://joagzb.com)
- [LinkedIn Nicolas](https://www.linkedin.com/in/nicolás-giuliano-204a301a4/)

## License

- [GPLv3](./LICENSE)
