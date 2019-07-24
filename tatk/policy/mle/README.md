# Imitation

Vanilla MLE Policy employs a multi-class classification via Imitation Learning with a set of compositional actions where a compositional action consists of a set of dialog act items.

## Train

```python
cd [dataset]
python train.py
```

You can modify *config.json* to change the setting.

## Data

data/[dataset]/[train|val|test].json.zip
