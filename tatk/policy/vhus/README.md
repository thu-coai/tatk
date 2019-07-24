# VHUS

A data driven variational hierarchical seq2seq user simulator where an unobserved latent random variable generates the user turn sequence.

## Train

```python
cd [dataset]
python train.py
```

You can modify *config.json* to change the setting.

## Data

data/camrest/CamRest676_v2.json

data/multiwoz/annotated_user_da_with_span_full.json

## Reference

```
@inproceedings{gur2018user,
  title={User modeling for task oriented dialogues},
  author={G{\"u}r, Izzeddin and Hakkani-T{\"u}r, Dilek and T{\"u}r, Gokhan and Shah, Pararth},
  booktitle={2018 IEEE Spoken Language Technology Workshop (SLT)},
  pages={900--906},
  year={2018},
  organization={IEEE}
}
```