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

clean:
	rm -rf ~/hacking-medium-headlines/data
	mkdir -p $(PROJECT_HOME)/data/raw
	mkdir -p $(PROJECT_HOME)/data/processed
	mkdir -p $(PROJECT_HOME)/data/final
	mkdir -p $(PROJECT_HOME)/models
	mkdir -p $(PROJECT_HOME)/reports

clean-data:
	rm -rf ~/hacking-medium-headlines/data
	mkdir -p $(PROJECT_HOME)/data/raw
	mkdir -p $(PROJECT_HOME)/data/processed
	mkdir -p $(PROJECT_HOME)/data/final

clean-models:
	rm -rf ~/hacking-medium-headlines/models
	mkdir -p $(PROJECT_HOME)/models
