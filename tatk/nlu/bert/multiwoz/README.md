# BERTNLU on multiwoz

Based on pre-trained bert, BERTNLU use a linear layer for slot tagging and another linear layer for intent classification. Dialog acts are split into two groups, depending on whether the value is in the utterance. 

- For those dialog acts that the value appears in the utterance, they are translated to BIO tags. For example, `"Find me a cheap hotel"`, its dialog act is `{"Hotel-Inform":[["Price", "cheap"]]}`, and translated tag sequence is `["O", "O", "O", "B-Hotel-Inform+Price", "O"]`. A linear layer takes pre-trained bert word embeddings as input and classify the tag label.
- For each of the other dialog acts, such as `(Hotel-Request, Address, ?)`, another linear layer takes pre-trained bert embeddings of `[CLS]` as input and do the binary classification.

## Example usage

Determine which data you want to use: if **mode**='usr', use user utterances to train; if **mode**='sys', use system utterances to train; if **mode**='all', use both user and system utterances to train.

#### Preprocess data

On `bert/multiwoz` dir:

```sh
$ python preprocess.py [mode]
```

output processed data on `data/[mode]_data/` dir.

#### Train a model

On `bert` dir:

```sh
$ PYTHONPATH=../../.. python train.py --config_path multiwoz/configs/multiwoz_[mode].json
```

The model will be saved on `output/[mode]/bestcheckpoint.tar`. Also, it will be zipped as `output/[mode]/bert_multiwoz_[mode].zip`. 

Trained models can be download on: 

- Trained on all data: [mode=all](https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_all.zip)
- Trained on user utterances only: [mode=usr](https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_usr.zip)
- Trained on system utterances only: [mode=sys](https://tatk-data.s3-ap-northeast-1.amazonaws.com/bert_multiwoz_usr.zip)

#### Evaluate

On `bert/multiwoz` dir:

```sh
$ PYTHONPATH=../../../.. python evaluate.py [mode]
```

#### Predict

In `nlu.py` , the `BERTNLU` class inherits the NLU interface and adapts to multiwoz dataset. Example usage:

```python
from tatk.nlu.bert.multiwoz import BERTNLU

model = BERTNLU(mode, model_file=PATH_TO_ZIPPED_MODEL)
dialog_act = model.predict(utterance)
```

You can refer to `evaluate.py` for specific usage.

## Data

We use the multiwoz data (`data/multiwoz/[train|val|test].json.zip`).

## Performance

`mode` determines the data we use: if mode=`usr`, use user utterances to train; if mode=`sys`, use system utterances to train; if mode=`all`, use both user and system utterances to train.

We evaluate the precision/recall/f1 of predicted dialog act.

| mode | Precision | Recall | F1    |
| ---- | --------- | ------ | ----- |
| usr  | 78.30     | 66.12  | 71.69 |
| sys  | 73.06     | 62.71  | 67.49 |
| all  | 71.92     | 60.30  | 65.60 |

## References

```
@inproceedings{devlin2019bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={4171--4186},
  year={2019}
}

@article{chen2019bert,
  title={BERT for Joint Intent Classification and Slot Filling},
  author={Chen, Qian and Zhuo, Zhu and Wang, Wen},
  journal={arXiv preprint arXiv:1902.10909},
  year={2019}
}
```

