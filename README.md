# Task-oriented Dialog System Toolkits
[![Build Status](https://travis-ci.com/thu-coai/tatk.svg?branch=master)](https://travis-ci.com/thu-coai/tatk) 

TaTk is an open-source task-oriented dialog system toolkits developed by Tsinghua University Conversational AI group (THU-coai). We provide several models for each module in dialog system, as well as some joint models and end-to-end models. It's easy to combine the modules to build a dialog system and replace some modules with yours to evaluate them in system level. Further more, user simulator (policy for user agent) is provided for system policy training. Our unified agent definition also supports symmetric agents for negotiation dialog and multiple agents for multiparty dialog.

Features included:

- Complete and configurable framework for task-oriented dialog system.
- Pre-trained models on Multiwoz, Camrest, Dealornot dataset.
- Simple interfaces for adapting your models.
- Rule simulators on Multiwoz and Camrest dataset for RL policy training.
- Unified agent definition which allows customized dialog scene such as multiparty dialog.

This project is a part of ``thutk`` (Toolkits for Dialog System by Tsinghua University), you can follow [thutk](http://coai.cs.tsinghua.edu.cn/thutk/) or 
[tatk](http://coai.cs.tsinghua.edu.cn/thutk/tatk/) on our home page. Some code are shared with [Convlab](https://github.com/ConvLab/ConvLab).

- [Installation](#installation)
- [Tutorials](#tutorials)
- [Documents](#documents)
- [Models](#models)
- [Supported Dataset](#Supported-Dataset)
- [Issues](#issues)
- [Contributions](#contributions)
- [Team](#team)
- [License](#license)

## Installation

Require python 3.6.

Clone this repository:
```bash
git clone https://github.com/thu-coai/tatk.git
```

Install tatk via pip:

```bash
cd tatk
pip install -e .
```

## Tutorials

Tutorials are under [tutorials](https://github.com/thu-coai/tatk/tree/master/tutorials) directory. You can also view it on [thutk/tatk](http://coai.cs.tsinghua.edu.cn/thutk/tatk/).

- [Getting Started](https://github.com/thu-coai/tatk/blob/master/tutorials/Getting_Started/Getting_Started.ipynb) (Have a try on [Colab](https://colab.research.google.com/github/thu-coai/tatk/blob/master/tutorials/Getting_Started/Getting_Started.ipynb)!)
- [Add New Model](https://github.com/thu-coai/tatk/blob/master/tutorials/Adapt_Models_and_Datasets/Add_New_Model.md)
- [RNN rollout - deal or not](https://github.com/thu-coai/tatk/blob/master/tutorials/Deal_or_Not/rnn_rollout_deal_or_not.md)
- [Train RL Policies](https://github.com/thu-coai/tatk/blob/master/tutorials/Train_RL_Policies/README.md)

## Documents

Our documents are on https://thu-coai.github.io/tatk_docs/.

## Models

We provide following models:

- NLU: SVMNLU, BERTNLU
- DST: rule, MDBT
- Policy: rule, Imitation, REINFORCE, PPO, MDRG
- Simulator policy: Agenda, VHUS
- NLG: Template, SCLSTM
- End2End: Sequicity, RNN_rollout

For  more details about these models, You can refer to `README.md` under `tatk/$module/$model/$dataset` dir such as [tatk/nlu/bert/multiwoz/README.md](https://github.com/thu-coai/tatk/blob/master/tatk/nlu/bert/multiwoz/README.md).

## Supported Dataset

- [Multiwoz](https://www.repository.cam.ac.uk/handle/1810/280608)
  - We add user dialog act (*inform*, *request*, *bye*, *greet*, *thank*), remove 5 sessions that have incomplete dialog act annotation and place it under `data/multiwoz` dir.
  - Train/val/test size: 8434/999/1000. Split as original data.
  - LICENSE: Attribution 4.0 International, url: http://creativecommons.org/licenses/by/4.0/
- [Camrest](https://www.repository.cam.ac.uk/handle/1810/260970)
  - We add system dialog act (*inform*, *request*, *nooffer*) and place it under `data/camrest` dir.
  - Train/val/test size: 406/135/135. Split as original data.
  - LICENSE: Attribution 4.0 International, url: http://creativecommons.org/licenses/by/4.0/
- [Dealornot](https://github.com/facebookresearch/end-to-end-negotiator/tree/master/src/data/negotiate)
  - Place it under `data/dealornot` dir.
  - Train/val/test size: 5048/234/526. Split as original data.
  - LICENSE: Attribution-NonCommercial 4.0 International, url: https://creativecommons.org/licenses/by-nc/4.0/

## Issues

You are welcome to create an issue if you want to request a feature, report a bug or ask a general question.

## Contributions

We welcome contributions from community.

- If you want to make a big change, we recommend first creating an issue with your design.
- Small contributions can be directly made by a pull request.
- If you like make contributions for our library, see issues to find what we need.

## Team

`tatk` is maintained and developed by Tsinghua university conversational AI group (THU-coai). Check our [main pages](http://coai.cs.tsinghua.edu.cn/) (In Chinese).

## License

Apache License 2.0
