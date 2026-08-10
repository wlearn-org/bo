NPM ?= npm
CMAKE ?= $(shell command -v cmake 2>/dev/null || echo cmake)
WLEARN_PYTHON ?= python
PYTHON ?= $(WLEARN_PYTHON)
PIP ?= $(PYTHON) -m pip
export WLEARN_PYTHON

.PHONY: sync-js-csrc sync-py-csrc build-c test-c build-js test-js build-py test-py wheel npm-pack test

sync-js-csrc:
	node js/scripts/sync-csrc.js

sync-py-csrc:
	$(PYTHON) py/scripts/sync-csrc.py

build-c:
	$(CMAKE) -S . -B build -DBUILD_TESTING=ON
	$(CMAKE) --build build

test-c: build-c
	./build/test_bo

build-js: sync-js-csrc
	cd js && $(NPM) run build
	cd js && $(NPM) run build:browser

test-js: build-js
	cd js && $(NPM) test

build-py: sync-py-csrc
	cd py && $(PYTHON) setup.py build_ext --inplace

test-py: build-py build-c
	PYTHONPATH=py $(PYTHON) test/test_python.py

wheel: sync-py-csrc
	mkdir -p build/wheelhouse
	cd py && $(PIP) wheel . -w ../build/wheelhouse --no-deps --no-build-isolation

npm-pack: build-js
	cd js && npm_config_cache=/tmp/npm-pack-audit $(NPM) pack --dry-run

test: test-c test-js test-py
