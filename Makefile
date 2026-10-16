VENV := .venv
PY := $(VENV)/bin/python

.PHONY: install run test lint

install:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

run:
	$(PY) scripts/run_experiments.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
