.PHONY: install test

install:
	pip install -r requirements.txt

test:
	docker build -f test.dockerfile -t quay.io/openai/gym:test .

upload:
	rm -rf dist
	python setup.py sdist
	twine upload dist/*

docker-build:
	docker build -t quay.io/openai/gym .

docker-run:
	docker run -ti quay.io/openai/gym bash
