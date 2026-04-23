# Availability Experiment (Python + Gurobi)

This folder is a Python port of the Julia availability experiment workflow.

## Included in this phase

- `main_availability.py`
- `availability.py`
- `parsers.py`
- `util.py`
- `simulation.py`
- `Algorithms/teavar.py`
- `data/B4` (copied from the Julia project)

## Run

From this directory:
```bash
cd e:\SIG\teavar\code_py
```

```bash
python main_availability.py availability
```

Optional plotting:

```bash
python main_availability.py availability -p
```

## Dependencies

- `gurobipy`
- `numpy`
- `networkx`
- `matplotlib` (optional, only when plotting)

安装依赖：
```bash
python -m pip install -r requirements.txt
```