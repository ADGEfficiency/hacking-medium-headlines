.PHONY: requirements

PROJECT_NAME = hacking-medium-headlines
PROJECT_HOME = $(HOME)/$(PROJECT_NAME)
PYTHON_INTERPRETER = python3

requirements:
	$(PYTHON_INTERPRETER) -m pip install --upgrade pip
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

dotenv:
	echo "PROJECT_NAME=$(PROJECT_NAME)" > .env
	echo "PROJECT_HOME=$(PROJECT_HOME)" > .env

init:
	mkdir -p $(PROJECT_HOME)/data/raw
	mkdir -p $(PROJECT_HOME)/data/interim
	mkdir -p $(PROJECT_HOME)/data/processed
	mkdir -p $(PROJECT_HOME)/models
	mkdir -p $(PROJECT_HOME)/reports

lint:
	flake8 src

clean_up:
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*.ipynb_checkpoints" -delete
	find . -type d -name "__pycache__" -delete
