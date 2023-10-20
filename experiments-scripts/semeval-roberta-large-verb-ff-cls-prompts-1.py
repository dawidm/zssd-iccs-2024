import itertools

from train import repeat_test_model

import logging
logging.getLogger().setLevel(logging.INFO)
handler = logging.StreamHandler()
logging.getLogger().addHandler(handler)

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

base_hparams = {
    'masked_lm': True,
    'clf_gelu_head': True,
    'masked_lm_verbalizer': False,
    'clf_hidden_dim': 1024,
    'clf_drop_prob': 0.1,
    'ft_epochs': 6,
    'lr_ft': 1e-5,
    'train_ft_batch': 32,
    'ft_eval_steps': 50,
    'warmup_steps_ft': 100,
    'seq_max_len': 85,
}
from sklearn.model_selection import ParameterGrid

hparams_list_1 = list(ParameterGrid({'use_semeval_zeroshot': ['DT','HC','AT','DT','FE','CL','AB'], 'masked_lm_prompt': [4, 10]}))

hparams_list_2 = [
    {'clf_gelu_head': False,
     'masked_lm': True,
     'masked_lm_verbalizer': True,},
    {'clf_gelu_head': True,
     'masked_lm': True, },
    {'clf_gelu_head': True,
     'masked_lm': False,
     'use_cls_token': True, }
]

hparams_list = [dict(**x[0], **x[1]) for x in list(itertools.product(hparams_list_1, hparams_list_2))]

for hparams in hparams_list:
    hparams.update(base_hparams)
    model_name = f'semeval-roberta-large-verb-ff-cls-prompts-1-{hparams["use_semeval_zeroshot"]}'
    repeat_test_model(10, 'roberta-large', model_name, hparams, finetune_only=True, report_to='wandb')