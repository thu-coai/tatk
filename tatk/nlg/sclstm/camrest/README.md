# nlg-sclstm-camrest

## Prepare the data

unzip [zip](https://tatk-data.s3-ap-northeast-1.amazonaws.com/nlg_sclstm_camrest.zip) here

## Data

We use the camrest data (./resource/\*, ./resource_usr/\*).

## Performance on Camrest

run `python evaluate.py [usr|sys|all]`

|       | usr    | sys    | all    |
| ----- | ------ | ------ | ------ |
| BLEU4 | 0.6507 | 0.5933 | 0.6156 |

