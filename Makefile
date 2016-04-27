.PHONY: install test

install:
	pip install -r requirements.txt

test:
	nose2

upload:
	rm -rf dist
	python setup.py sdist
	twine upload dist/*
