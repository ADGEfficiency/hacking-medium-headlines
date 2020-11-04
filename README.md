# hacking-medium-headlines

Predicting claps from headlines on Medium.

## Setup

Due to the wide variety in options for managing virtual environments, we leave it up to the user to create and activate your virtual environment.

```bash
$ make requirements; make dotenv; make init;
```

## Use

```bash
$ python3 src/archive.py
```

Test:

```bash
$ pytest src/tests.py
```
