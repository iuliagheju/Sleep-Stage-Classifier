VENV?=.venv
PYTHON?=$(VENV)/Scripts/python.exe
PIP?=$(VENV)/Scripts/pip.exe

.PHONY: install test smoke lint format

install:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]

lint:
	$(PYTHON) -m black src tests
	$(PYTHON) -m isort src tests
	$(PYTHON) -m mypy src

format:
	$(PYTHON) -m black src tests
	$(PYTHON) -m isort src tests

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -m pytest -q -m smoke
