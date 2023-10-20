# Source code for *Target-phrase Zero-shot Stance Detection: Where Do We Stand?*

## Requirements

python==3.8.10

more: requirements.txt

## Our experiments
Our experiments scripts are in folder `experiments-scripts`. Details:

* Sem2016T6 BERT-base experiments: `semeval-bert-base-verb-vs-ff-vs-cls-1.py`, `allaway-seval-bert-baseline-1.py`
* Sem2016T6 BERT-large experiments: `semeval-bert-large-cls-1.py`
* Sem2016T6 ROBERTA-base experiments: `semeval-roberta-base-verb-vs-ff-vs-cls-1.py`
* Sem2016T6 ROBERTA-large experiments: `semeval-roberta-large-verb-ff-cls-prompts-1.py`
* VAST BERT-base experiments: `vast-bert-base-1.py`, `vast-bert-base-2.py`, `vast-bert-base-3.py`
* VAST BERT-base prompts experiments`vast-bert-base-ff-vs-verb-1.py`
* VAST BERT-large experiments: `vast-bert-large-1.py`
* VAST RoBERTa-base experiments: `vast-roberta-base-1.py`
* VAST RoBERTa-large experiments: `vast-roberta-large-1.py`

## Reproduced experiments
Code changes required for reproducing experiments are in folders: `JointCL`, `VTCG`, `WS-BERT`, `BS-RGCN`.

## Datasets files
Datasets should be put in `data` directory with following filenames:
* Sem2016T6: `se_train.csv`, `se_test.csv`
* VAST: `vast_train.csv`, `vast_dev.csv`, `vast_test.csv`. `datasets-processing/vast_topic_prep.py` should be run before running experiments.



