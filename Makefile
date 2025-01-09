VERSION := $(shell python3 -c "import re; print(re.search(r'version = (\"[^\"]+\")', open('pyproject.toml').read()).group(1))")
WHEEL_NAME := power-attention-$(VERSION)-cp311-cp311-manylinux_x86_64

# Fast build configuration
FAST_IS_EVEN_MN ?= true
FAST_IS_EVEN_K ?= true
FAST_DEG ?= 2
FAST_STATE_DEG ?= 2
FAST_GATING ?= true
FAST_FLASH_EQUIVALENT ?= false
FAST_NORMAL_SPACE ?= true
FAST_IS_CAUSAL ?= true
FAST_IS_FP16 ?= true
FAST_HEAD_DIM ?= 64

export FAST_IS_EVEN_MN
export FAST_IS_EVEN_K
export FAST_DEG
export FAST_GATING
export FAST_FLASH_EQUIVALENT
export FAST_NORMAL_SPACE
export FAST_IS_CAUSAL
export FAST_IS_FP16
export FAST_HEAD_DIM

.PHONY: build dev test benchmark install dev report

clean_dist:
	rm -rf dist/*

create_dist: clean_dist
	python setup.py sdist

upload_package: create_dist
	twine upload dist/*

build:
	python setup.py bdist_wheel

test:
	pytest test/test_all.py

benchmark:
	cd test && python benchmark.py

install:
	pip uninstall -y power-attention && pip install dist/$(WHEEL_NAME).whl

debug:
	DEBUG_POWER_BWD_DKDV=1 DEBUG_POWER_BWD_DQ=1 make build
	pip uninstall -y power-attention && pip install dist/$(WHEEL_NAME).debug.whl

dev:
	$(if $(filter-out dev,$(MAKECMDGOALS)),\
		$(MAKE) build && \
		cp dist/$(WHEEL_NAME).whl dist/$(WHEEL_NAME).$(filter-out dev,$(MAKECMDGOALS)).whl && \
		pip uninstall -y power-attention && pip install dist/$(WHEEL_NAME).$(filter-out dev,$(MAKECMDGOALS)).whl, \
		$(MAKE) build install)

report:
	python power_attention/_attention/impl.py
	@echo "\n"
	python power_attention/_update_state/impl.py
	@echo "\n"
	python power_attention/_query_state/impl.py
	@echo "\n"
	python power_attention/power_full.py
	@echo "\n"
	cd ../../ && python -m measurements.cudoc -k ".*wd.*" -x "convert" python power_attention/power_attention/power_full.py profile && cd packages/power_attention

fast:
	FAST_BUILD=1 python setup.py bdist_wheel
	pip uninstall -y power-attention && pip install dist/$(WHEEL_NAME).whl

fast-verbose:
	FAST_BUILD=1 NVCC_VERBOSE=1 python setup.py bdist_wheel
	pip uninstall -y power-attention && pip install dist/$(WHEEL_NAME).whl

%:
	@:
