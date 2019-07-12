# RNN Rollout model for Object Division Negotiation Dialog

## Data preparation
To use this model, you have to first download the pre-trained models
from [here](s3://tatk-data/rnnrollout_dealornot.zip), and put the *.th
files under tatk/e2e/rnn_rullout/deal_or_not/configs/.

## Run the Model
To run the model, you can run this command:
```
python test_deal_or_not.py
```
under tests/e2e/rnn_rollout directory.