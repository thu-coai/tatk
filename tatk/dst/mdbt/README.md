# Multi-domain Belief Tracker
The multidomain belief tracker (MDBT) is a belief tracking model that
fully utilizes semantic similarity between dialogue utterances and the
ontology terms, which is proposed by [Ramadan et al., 2018](https://www.aclweb.org/anthology/P18-2069).

## Package Structure
We adopted the original code to make it a flexible module which can be
easily imported in a pipeline dialog framework. The dataset-independent
implementation for MDBT is in ```tatk/dst/mdbt```, and that for Multiwoz
dataset is in ```tatk/dst/mdbt/multiwoz```.

## Run the Code
To run the code, you have to download the pre-trained model and data
from [here](https://drive.google.com/open?id=1k6wbabIlYju7kR0Zr4aVXwE_fsGBOtdw),
and put the ```word-vectors, models``` and ```data``` directories under
```tatk/dst/mdbt/multiwoz/configs```.