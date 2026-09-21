# Urban Ants

A route-finding experiment between two points in a city using walking and urban buses. Edge weights form one static generalized-cost score; they are not calibrated travel minutes or converted fares, and the model does not represent schedules or headways.

More details about the methodology can be found at: [Joaquin's portfolio](https://joagzb.com)

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
   python -m src.main
   ```
   The equivalent command from inside `src/` is `python -m main`.

The demo writes `aco_route.html`, `acs_route.html`, and `bwas_route.html` to the repository-root `tmp/` directory. Each interactive HTML title and route legend identifies the algorithm and configured ant count.

## Notebook experiment

Open `src/TPF.ipynb` from either the repository root or `src/` after activating the project environment. The notebook does not rewrite import or output roots. It loads algorithm-specific profiles from `src/configuration/algorithm_settings.py`; use the lightweight CLI above for the base 20-ant, 100-epoch settings.

Visual results are generated under the repository-root `tmp/` directory. The notebook records each algorithm's JSONL history and writes its interactive animation there.

## Running tests

After installing the dependencies, this step is required for running the tests

```bash
pip install pytest
```

you can run the test suite with [pytest](https://docs.pytest.org/en/stable/) by directly running:

```bash
pytest
```

## Authors

- [LinkedIn Joaquin](https://www.linkedin.com/in/joaquin-gonzalez-budino/)
- [LinkedIn Nicolas](https://www.linkedin.com/in/nicolás-giuliano-204a301a4/)

## License

- [GPLv3](./LICENSE)
