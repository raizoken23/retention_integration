# Get version from pyproject.toml
VERSION := $(shell python scripts/get_version.py)
PACKAGE_NAME := power-attention
<<<<<<< HEAD

# Find Python 3.11+
PYTHON := $(shell for py in python3.12 python3.11 python3 python; do \
    if command -v $$py >/dev/null && $$py --version 2>&1 | grep -q "Python 3.1[1-9]"; then \
        echo $$py; \
        break; \
    fi \
done)

ifeq ($(PYTHON),)
    $(error Python 3.11 or higher is required. Please install Python 3.11+)
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
=======

.PHONY: clean check-version check-test-version release release-test help
>>>>>>> a7a60ad (API Reference complete)

# Clean and check
clean:
	rm -rf dist/ build/ *.egg-info/ *.so wheelhouse/

# Version checking
check-version:
	@echo "Local version: $(VERSION)"
	@python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)"

check-test-version:
	@echo "Local version: $(VERSION)"
	@python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)" --test

# Release commands
release: clean check-version
	@echo "Building wheels with cibuildwheel..."
	python -m cibuildwheel --output-dir dist
	python -m twine check dist/*
	@echo "Uploading to PyPI..."
	python -m twine upload dist/*
	@echo "Release $(VERSION) completed!"

release-test: clean check-test-version
	@echo "Building wheels with cibuildwheel..."
	python -m cibuildwheel --output-dir dist
	python -m twine check dist/*
	@echo "Uploading to TestPyPI..."
	python -m twine upload --repository testpypi dist/*
	@echo "Test release $(VERSION) completed!"

# Help
help:
	@echo "Available commands:"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make release       - Release to PyPI (includes version check)"
	@echo "  make release-test  - Release to TestPyPI"
	@echo "  make check-version - Check version against PyPI"
	@echo "  make check-test-version - Check version against TestPyPI"
	@echo "Current version: $(VERSION)" 
