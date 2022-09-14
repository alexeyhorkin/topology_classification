# Topology classification
The repository contains сode for paper - TBA

![demo image](resources/picture_from_paper.png)

## Instalation

### Linux (preferred)
1. Install [poetry](https://python-poetry.org/)

2. Install deps:

```bash
poetry install
```

### MacOS (arm)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_mac_arm.txt
```

## Run notebooks:

On Linux:
```bash
poetry run jupyter-notebook notebooks
```

On MacOS:
```bash
jupyter-notebook notebooks
```

