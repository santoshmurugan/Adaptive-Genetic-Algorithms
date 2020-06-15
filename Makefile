.PHONY: clean-pyc clean-build docs clean

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "test-all - run tests on every Python version with tox"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "release - package and upload a release"
	@echo "dist - package"

clean: clean-build clean-pyc clean-tox
	rm -fr htmlcov/

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.py-e' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

clean-tox:
	rm -fr *.egg

lint:
	flake8 pyeasyga tests examples

test:
	python setup.py test

test-all:
	tox

coverage:
	coverage run --source pyeasyga setup.py test
	coverage report -m
	coverage html
	open htmlcov/index.html

docs:
	rm -f docs/pyeasyga.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ pyeasyga
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	open docs/_build/html/index.html

release: clean
	python setup.py sdist --formats=gztar,zip upload -r pypi
	python setup.py upload_docs -r pypi
	python setup.py bdist_wheel upload -r pypi

dist: clean
	python setup.py sdist --formats=gztar,zip
	python setup.py bdist_wheel
	ls -l dist
