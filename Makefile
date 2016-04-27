.PHONY: install test

install:
	pip install -r requirements.txt

test:
	nose2
