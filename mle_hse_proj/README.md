# Installation

In this project you should use uv.
uv installation:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

init uv and create venv

```
uv init
uv venv
```

# Requeirements
see the requirements.txt

installation:

```
uv pip install -r requirements.txt
```

```sample usage

uv run pipeline.py --feature_columns Pclass Sex Age SibSp Parch Fare Embarked --target_column Survived --model svm --model_kwargs C=0.1 --train_csv_path train.csv

```