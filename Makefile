# Get version from pyproject.toml
VERSION := $(shell python scripts/get_version.py)
PACKAGE_NAME := power-attention

# Get required Python version from pyproject.toml
REQUIRED_PYTHON_VERSION := $(shell python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['requires-python'].replace('>=',''))")

# Function to check Python version - store result in a single word
define check_python_version
$(shell command -v $(1) > /dev/null && $(1) -c "import sys; exit(0 if sys.version_info >= tuple(map(int, '$(REQUIRED_PYTHON_VERSION)'.split('.'))) else 1)" 2>/dev/null && echo "$(1)" || echo "")
endef

# Try to find a suitable Python interpreter
PYTHON_CANDIDATES := python3.12 python3.11 python3 python
PYTHON := $(strip $(firstword $(foreach py,$(PYTHON_CANDIDATES),$(call check_python_version,$(py)))))

ifeq ($(PYTHON),)
    $(error No Python $(REQUIRED_PYTHON_VERSION)+ interpreter found. Please install Python $(REQUIRED_PYTHON_VERSION) or higher)
endif

# Allow overriding venv path through environment variable, default to .venv
VENV_DIR ?= $(if $(POWER_ATTENTION_VENV_PATH),$(POWER_ATTENTION_VENV_PATH),.venv)
PIP := $(VENV_DIR)/bin/pip
PYTEST := $(VENV_DIR)/bin/pytest

.PHONY: all build test benchmark release release-test clean check-version venv deps deps-dev deps-benchmark deps-train

all: build test

dev: venv
	CC=gcc CXX=g++ $(PIP) install -e .

venv:
	@echo "Creating virtual environment using $(PYTHON) ($(shell $(PYTHON) --version 2>&1))"
	$(PYTHON) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip

deps: venv
	$(PIP) install -r requirements.txt

deps-dev: deps
	$(PIP) install -r requirements-dev.txt

deps-benchmark: deps
	$(PIP) install -r requirements-benchmark.txt

deps-train: deps
	$(PIP) install -r requirements-train.txt

# Development commands
test: deps-dev
	$(PYTEST) power_attention/tests.py -v

benchmark: deps-benchmark
	$(VENV_DIR)/bin/python test/benchmark.py

# Version checking
check-version:
	@echo "Local version: $(VERSION)"
	@$(VENV_DIR)/bin/python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)"

check-test-version:
	@echo "Local version: $(VERSION)"
	@$(VENV_DIR)/bin/python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)" --test

# Clean and check
clean:
	rm -rf dist/ build/ *.egg-info/ *.so wheelhouse/

# Release commands
release: clean deps-dev check-version
	@echo "Building wheels with cibuildwheel..."
	$(VENV_DIR)/bin/python -m cibuildwheel --output-dir dist
	$(VENV_DIR)/bin/twine check dist/*
	@echo "Uploading to PyPI..."
	$(VENV_DIR)/bin/twine upload dist/*
	@echo "Release $(VERSION) completed!"

release-test: clean deps-dev check-test-version
	@echo "Building wheels with cibuildwheel..."
	$(VENV_DIR)/bin/python -m cibuildwheel --output-dir dist
	$(VENV_DIR)/bin/twine check dist/*
	@echo "Uploading to TestPyPI..."
	$(VENV_DIR)/bin/twine upload --repository testpypi dist/*
	@echo "Test release $(VERSION) completed!"

# Help
help:
	@echo "Available commands:"
	@echo "Environment variables:"
	@echo "  POWER_ATTENTION_VENV_PATH - Override default virtualenv location (.venv)"
	@echo ""
	@echo "Commands:"
	@echo "  make venv          - Create virtual environment"
	@echo "  make deps          - Install base dependencies"
	@echo "  make deps-dev      - Install development dependencies"
	@echo "  make deps-benchmark - Install benchmark dependencies"
	@echo "  make deps-train    - Install training dependencies"
	@echo "  make test          - Run tests"
	@echo "  make benchmark     - Run benchmarks"
	@echo "  make build         - Build package"
	@echo "  make fast          - Quick build for development"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make release       - Release to PyPI (includes version check)"
	@echo "  make release-test  - Release to TestPyPI"
	@echo "  make check-version - Check version against PyPI"
	@echo "  make check-test-version - Check version against TestPyPI"
	@echo "Current version: $(VERSION)" 
