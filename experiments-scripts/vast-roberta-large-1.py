from train import repeat_test_model

import logging
logging.getLogger().setLevel(logging.INFO)
handler = logging.StreamHandler()
logging.getLogger().addHandler(handler)

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

base_hparams = {
    'use_vast': True,
    'vast_filter_ambiguous': True,
    'vast_filter_type_3': True,
    'masked_lm': True,
    'clf_hidden_dim': 1024,
    'clf_drop_prob': 0.1,
    'ft_epochs': 6,
    'lr_ft': 1e-5,
    'train_ft_batch': 32,
    'ft_eval_steps': 100,
    'warmup_steps_ft': 100,
    'seq_max_len': 282,
}

hparams_list = [
    {'masked_lm_verbalizer': True,
     'clf_gelu_head': False,
     'masked_lm_prompt': 4,},
    {'masked_lm_verbalizer': False,
     'clf_gelu_head': True,
     'masked_lm_prompt': 4,},
    {'masked_lm': False,
     'use_cls_token': True,
     'clf_gelu_head': True,
     'masked_lm_prompt': 4,},

]

for hparams in hparams_list:
    hparams.update(base_hparams)
    model_name = f'vast-roberta-large-1'
    repeat_test_model(10, 'roberta-large', model_name, hparams, finetune_only=True, report_to='wandb')
