.PHONY: clean clean-build clean-pyc

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-cache ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-cache:  ## remove additional cached files
	rm -fr .mypy_cache
	rm -fr .pytest_cache
	rm -fr .tox
	rm -fr .coverage

format: ## linting, type checking, <> is replaced with package name
	poetry run black surrogate_data tests
	poetry run isort surrogate_data tests
	poetry run mypy surrogate_data tests

test: ## run test suite
	poetry run pytest

tox: ## run tox
	poetry run tox -p

tox-fresh: ## run tox with fresh environments
	poetry run tox -p -r

requirements: ## freeze requirements
	poetry export -f requirements.txt > requirements.txt

dist: clean ## builds source and wheel package
	python -m build

release_test:
	twine upload --repository testpypi dist/*

release: dist ## package and upload a release (requires the twine package)
	twine upload dist/*
