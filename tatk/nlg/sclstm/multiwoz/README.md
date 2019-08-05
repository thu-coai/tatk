# nlg-sclstm-multiwoz

## Prepare the data

unzip [zip](https://tatk-data.s3-ap-northeast-1.amazonaws.com/nlg_sclstm_multiwoz.zip) here

## Data

We use the multiwoz data (./resource/\*, ./resource_usr/\*).

## Performance on Multiwoz

run `python evaluate.py [usr|sys|all]`

|       | usr    | sys    | all    |
| ----- | ------ | ------ | ------ |
| BLEU4 | 0.7432 | 0.4885 | 0.3983 |

