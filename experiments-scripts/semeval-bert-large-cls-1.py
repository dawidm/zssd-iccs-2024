from train import repeat_test_model

import logging
logging.getLogger().setLevel(logging.INFO)
handler = logging.StreamHandler()
logging.getLogger().addHandler(handler)

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

base_hparams = {
    'use_bert_pooling_out': True,
    'clf_hidden_dim': 1024,
    'clf_drop_prob': 0.1,
    'ft_epochs': 6,
    'lr_ft': 1e-5,
    'train_ft_batch': 32,
    'ft_eval_steps': 100,
    'warmup_steps_ft': 100,
    'seq_max_len': 85,
}

hparams_list = [
    {'use_semeval_zeroshot': 'AT', },
    {'use_semeval_zeroshot': 'DT', },
    {'use_semeval_zeroshot': 'HC', },
    {'use_semeval_zeroshot': 'CL', },
    {'use_semeval_zeroshot': 'AB', },
    {'use_semeval_zeroshot': 'FE', },
]

for hparams in hparams_list:
    hparams.update(base_hparams)
    model_name = f'semeval-bert-large-cls-1'
    repeat_test_model(10, 'bert-large-uncased', model_name, hparams, finetune_only=True, report_to='wandb')
