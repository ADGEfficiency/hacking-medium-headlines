# hacking-medium-headlines

Predicting claps from headlines on Medium.


## Setup

Due to the wide variety in options for managing virtual environments, we leave it up to the user to create and activate your virtual environment.

First run to setup Python packages & folders:

```bash
$ make requirements; make dotenv; make init;
```

Download data & model artifacts into `~/hacking-medium-headlines` from S3:

```bash
make pulls3
```


## Use

### CLI to predict claps from headline

```bash
python3 cli.py "HEADLINE" "SITE_ID"
```

For example:

```bash
python3 cli.py "Four Ways to Make Millions on Medium" "towardsdatascience"
```

See available sites - including their `SITE_ID`:

```bash
$ python3 inspect.py
```

### Reproduce the data science work from scratch

Download data into `~/hacking-medium-headlines/data/raw`:

```bash
$ python3 src/scrape.py
```

Run grid searches and train final models:

```bash
$ python3 src/ml.py
```

Also included are notebooks documenting part of the process
- `notebooks/eda.ipynb` - initial data analysis
- `notebooks/error-analysis.ipynb` - looking for patterns in the errors made by the model


### Test

```bash
$ pytest src/tests.py
```
