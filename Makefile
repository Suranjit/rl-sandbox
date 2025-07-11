# Makefile for the RL Playground Project

# --- Configuration ---
# Grabs the current folder name to use for the virtual environment.
VENV_NAME := $(shell basename $(CURDIR))_env
PYTHON := $(VENV_NAME)/bin/python

# Default values for run commands, can be overridden from the command line.
# e.g., make run-train STRATEGY=dense ITERS=100
STRATEGY ?= simple
ITERS ?= 10
VIDEO ?= false

# --- Setup and Installation ---

.PHONY: setup
setup: $(VENV_NAME)/bin/activate

# Rule to create the virtual environment and install dependencies.
$(VENV_NAME)/bin/activate: pyproject.toml
	@echo "--- Creating virtual environment: $(VENV_NAME) ---"
	python3 -m venv $(VENV_NAME)
	@echo "--- Installing dependencies from pyproject.toml ---"
	$(PYTHON) -m pip install --upgrade pip
	# Install the project in editable mode with the 'dev' optional dependencies
	$(PYTHON) -m pip install -e '.[dev]'
	@echo "\n✅ Setup complete. Activate with: source $(VENV_NAME)/bin/activate"

# --- Code Quality & Testing ---

.PHONY: lint test
lint: setup
	@echo "--- Formatting code with Ruff ---"
	$(PYTHON) -m ruff format .
	@echo "--- Checking and fixing linting errors with Ruff ---"
	$(PYTHON) -m ruff check --fix .

test: setup
	@echo "--- Running tests with pytest and generating coverage report ---"
	$(PYTHON) -m pytest --cov=pong --cov-report=term-missing --cov-report=html

# --- Core Commands ---

.PHONY: run-tune run-train-tuned run-train run-eval
run-tune: setup
	@echo "--- 🔍 Tuning hyperparameters for strategy: $(STRATEGY) ---"
	$(PYTHON) train_pong.py --mode tune --strategy $(STRATEGY)

run-train-tuned: setup
	@echo "--- 🏋️  Continuing training for tuned strategy: $(STRATEGY) for $(ITERS) iterations ---"
	$(PYTHON) train_pong.py --mode train_tuned --strategy $(STRATEGY) --iters $(ITERS)

run-train: setup
	@echo "--- 💪 Starting fresh training for strategy: $(STRATEGY) for $(ITERS) iterations ---"
	$(PYTHON) train_pong.py --mode train --strategy $(STRATEGY) --iters $(ITERS)

run-eval: setup
	@echo "--- 🎮 Evaluating strategy: $(STRATEGY) ---"
ifeq ($(VIDEO), true)
	$(PYTHON) train_pong.py --mode eval --strategy $(STRATEGY) --video
else
	$(PYTHON) train_pong.py --mode eval --strategy $(STRATEGY)
endif

# --- Utility Commands ---

.PHONY: clean
clean:
	@echo "--- Removing virtual environment and temp files ---"
	rm -rf $(VENV_NAME)
	rm -rf .pytest_cache
	rm -rf **/__pycache__
	rm -rf **/.pyc
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "✅ Cleanup complete."

.PHONY: help
help:
	@echo "Makefile for RL Playground"
	@echo ""
	@echo "Usage:"
	@echo "  make setup          - Creates the virtual environment and installs dependencies."
	@echo "  make lint           - Formats the code and checks for linting errors."
	@echo "  make test           - Runs tests and calculates code coverage."
	@echo "  make run-tune       - Run hyperparameter tuning. STRATEGY can be set."
	@echo "  make run-train      - Train a model from scratch. STRATEGY and ITERS can be set."
	@echo "  make run-train-tuned- Train from the best tuned model. STRATEGY and ITERS can be set."
	@echo "  make run-eval       - Evaluate the best model. Set STRATEGY and VIDEO=true to record."
	@echo "  make clean          - Deletes the virtual environment and temporary files."
	@echo ""
	@echo "Examples:"
	@echo "  make run-train STRATEGY=dense ITERS=50"
	@echo "  make run-eval STRATEGY=selfplay_dense VIDEO=true"