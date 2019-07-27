# Sequicity

Sequicity is an end-to-end task-oriented dialog system based on a single sequence-to-sequence model that uses *belief span* to track dialog believes. We adapt the code from [github](https://github.com/WING-NUS/sequicity) to work in multiwoz corpus. The original paper can be found at [ACL Anthology](https://aclweb.org/anthology/papers/P/P18/P18-1133).

## Prepare data

To prepare the data for Sequicity, you need to download data and unzip in `[dataset]` dir. Please refer to `[dataset]` subdirectory for details.

## Training with default parameters

```bash
$ PYTHONPATH=../../../../.. python model.py -mode train -model [dataset] -cfg [dataset]/configs/[dataset].json
$ PYTHONPATH=../../../../.. python model.py -mode adjust -model [dataset] -cfg [dataset]/configs/[dataset].json
```

## Testing

```bash
$ PYTHONPATH=../../../../.. python model.py -mode test -model [dataset] -cfg [dataset]/configs/[dataset].json
```

## Reinforcement fine-tuning

```bash
$ PYTHONPATH=../../../../.. python model.py -mode rl -model [dataset] -cfg [dataset]/configs/[dataset].json
```

## Trained model

Please refer to `[dataset]` subdirectory for details.

## Evaluation Metrics

- BLEU
- Match rate : determines if a system can generate all correct constraints (belief span) to search the indicated entities of the user
- Success F1: F1 score of requested slots answered in the current dialogue

## Predict

```python
from tatk.e2e.sequicity.[dataset] import Sequicity

s = Sequicity(model_file=MODEL_PATH_OR_URL)
s.response("I want to find a cheap restaurant")
```

## Reference

   ```
@inproceedings{lei2018sequicity,
	title={Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures},
	author={Lei, Wenqiang and Jin, Xisen and Ren, Zhaochun and He, Xiangnan and Kan, Min-Yen and Yin, Dawei},
	booktitle={ACL},
	year={2018}
}
   ```