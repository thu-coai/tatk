# Sequicity on multiwoz

Sequicity is an end-to-end task-oriented dialog system based on a single sequence-to-sequence model that uses *belief span* to track dialog believes. We adapt the code from [github](https://github.com/WING-NUS/sequicity) to work in multiwoz corpus.  The original paper can be found at [ACL Anthology](https://aclweb.org/anthology/papers/P/P18/P18-1133)

## Prepare data

To prepare the data for Sequicity, you need to download [data](https://tatk-data.s3-ap-northeast-1.amazonaws.com/sequicity_multiwoz_data.zip) and unzip in `sequicity/multiwoz` dir.

## Training with default parameters

Go to `sequicity` dir,

   ```bash
$ PYTHONPATH=../../../../.. python model.py -mode train -model multiwoz -cfg multiwoz/configs/multiwoz.json
$ PYTHONPATH=../../../../.. python model.py -mode adjust -model multiwoz -cfg multiwoz/configs/multiwoz.json
   ```

   ## Testing

   ```bash
$ PYTHONPATH=../../../../.. python model.py -mode test -model multiwoz -cfg multiwoz/configs/multiwoz.json
   ```

   ## Reinforcement fine-tuning

   ```bash
$ PYTHONPATH=../../../../.. python model.py -mode rl -model multiwoz -cfg multiwoz/configs/multiwoz.json
   ```

## Trained model

Trained model can be download on [here](https://tatk-data.s3-ap-northeast-1.amazonaws.com/sequicity_multiwoz.pkl). Place it under `sequicity/multiwoz/output` dir.

## Predict

```python
from tatk.e2e.sequicity.multiwoz import Sequicity

s = Sequicity(model_file=MODEL_PATH_OR_URL)
s.response("I want to find a cheap restaurant")
```

## Data

https://www.repository.cam.ac.uk/handle/1810/280608

## Reference

   ```
@inproceedings{lei2018sequicity,
	title={Sequicity: Simplifying Task-oriented Dialogue Systems with Single Sequence-to-Sequence Architectures},
	author={Lei, Wenqiang and Jin, Xisen and Ren, Zhaochun and He, Xiangnan and Kan, Min-Yen and Yin, Dawei},
	year={2018},
	organization={ACL}
}
   ```