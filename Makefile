# Get version from pyproject.toml
UV_RUN := uv run --no-project
VERSION := $(shell $(UV_RUN) python scripts/get_version.py)
PACKAGE_NAME := power-attention

.PHONY: all build test benchmark release release-test clean check-version

all: build test

dev:
	CC=gcc CXX=g++ uv sync

deps:
	uv sync --no-install-project --group dev --group train

# Development commands
test:
	$(UV_RUN) pytest power_attention/tests.py -v

benchmark:
	$(UV_RUN) python test/benchmark.py

# Build commands
build:
	CC=gcc CXX=g++ $(UV_RUN) python -m build

fast:
	CC=gcc CXX=g++ FAST_BUILD=1 $(UV_RUN) python -m build

# Version checking
check-version:
	@echo "Local version: $(VERSION)"
	@$(UV_RUN) python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)"

check-test-version:
	@echo "Local version: $(VERSION)"
	@$(UV_RUN) python scripts/version_check.py "$(VERSION)" "$(PACKAGE_NAME)" --test

# Clean and check
clean:
	rm -rf dist/ build/ *.egg-info/ *.so wheelhouse/

check-dist: build
	$(UV_RUN) twine check dist/*

# Release commands
release: clean check-version
	@echo "Building wheels with cibuildwheel..."
	$(UV_RUN) python -m cibuildwheel --output-dir dist
	@echo "Uploading to PyPI..."
	$(UV_RUN) twine upload dist/*
	@echo "Release $(VERSION) completed!"

release-test: clean check-test-version
	@echo "Building wheels with cibuildwheel..."
	$(UV_RUN) python -m cibuildwheel --output-dir dist
	@echo "Uploading to TestPyPI..."
	$(UV_RUN) twine upload --repository testpypi dist/*
	@echo "Test release $(VERSION) completed!"

# Help
help:
	@echo "Available commands:"
	@echo "  make deps          - Install development dependencies"
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
