# nlg-sclstm-multiwoz

## Prepare the data

unzip [zip](https://tatk-data.s3-ap-northeast-1.amazonaws.com/nlg_sclstm_multiwoz.zip) here

## Data

We use the multiwoz data (./resource/\*, ./resource_usr/\*).

## Evaluate

run `python evaluate.py [usr|sys|all]`

## Performance on Multiwoz

`mode` determines the data we use: if mode=`usr`, use user utterances to train; if mode=`sys`, use system utterances to train; if mode=`all`, use both user and system utterances to train.

We evaluate the BLEU4 of delexicalized utterance. The references of a generated sentence are all the golden sentences that have the same dialog act.

|       | usr    | sys    | all    |
| ----- | ------ | ------ | ------ |
| BLEU4 | 0.7432 | 0.4885 | 0.3983 |

