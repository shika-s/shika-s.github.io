# DATASCI207_Bird_Sounds
Local Copy of Team Project Repo for DATASCI 207

Team Members: Shikha Sharma, [Denvir Higgins](https://github.com/denvir-py), [WooJung Kim](https://github.com/WooJung-K), [David Lin](https://github.com/ddlin-mids) 

## Prerequisites

- pyenv (https://github.com/pyenv/pyenv)
- Python 3.10.x
- Poetry (https://python-poetry.org)

## Installation

### 1. Python via pyenv

```bash
pyenv install 3.10.12  # or latest 3.10.x
pyenv local 3.10.12
```

### 2. Poetry setup

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry config virtualenvs.in-project true
```

### 3. Install dependencies

```bash
poetry install
```

## Data

Download the BirdCLEF 2025 dataset from the Kaggle competition page:

https://www.kaggle.com/competitions/birdclef-2025/

Place the `birdclef-2025.zip` file into `data/raw` and then unzip:

```bash
cd data/raw
unzip birdclef-2025.zip -d .
```

Alternatively, using the Kaggle CLI:

```bash
kaggle competitions download -c birdclef-2025 -p data/raw
unzip data/raw/birdclef-2025.zip -d data/raw
```

## Usage

### Exploratory Data Analysis

Activate the virtual environment and launch Jupyter:

```bash
poetry shell
jupyter notebook
```

### Deep Learning Training

The project includes a comprehensive Keras-based training pipeline with SLURM support for hyperparameter optimization and distributed training.

