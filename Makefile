all: setup

.PHONY: requirements.txt test

setup:
	python3 -m venv ./
	. ./bin/activate
	pip install -r ./oci/requirements.txt

test:
	. ./bin/activate
	pytest ./test/

clean:
	. ./bin/activate
	rm -rf ./.pytest_cache
	rm -rf ./__pycache__
	deactivate

build:
	docker build --tag calculus-py:1.0 .

