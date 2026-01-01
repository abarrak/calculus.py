all: setup

.PHONY: requirements.txt test

setup:
	python3 -m venv ./venv
	. ./venv/bin/activate
	pip install -r ./requirements.txt

test:
	. ./venv/bin/activate
	pytest ./test/

clean:
	. ./venv/bin/activate
	rm -rf ./.pytest_cache
	rm -rf ./__pycache__
	deactivate

build:
	docker build --tag calculus-py:1.0 .
