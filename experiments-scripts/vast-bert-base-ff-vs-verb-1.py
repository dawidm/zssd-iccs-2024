from train import repeat_test_model

import logging
logging.getLogger().setLevel(logging.INFO)
handler = logging.StreamHandler()
logging.getLogger().addHandler(handler)

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

base_hparams = {
    'use_vast': True,
    'masked_lm': True,
    'masked_lm_prompt': 4,
    'clf_hidden_dim': 768,
    'clf_drop_prob': 0.1,
    'ft_epochs': 6,
    'lr_ft': 5e-5,
    'train_ft_batch': 32,
    'ft_eval_steps': 100,
    'warmup_steps_ft': 100,
    'seq_max_len': 180,
    'vast_preprocess_text': False,
    'vast_preprocess_topic': False,
    'vast_filter_ambiguous': True,
    'vast_filter_type_3': True
}

hparams_list = [
    {'masked_lm_verbalizer': True,
     'clf_gelu_head': False, },
    {'masked_lm_verbalizer': False,
     'clf_gelu_head': True, },
]

for hparams in hparams_list:
    hparams.update(base_hparams)
    model_name = f'vast-bert-base-ff-vs-verb-1'
    repeat_test_model(10, 'bert-base-uncased', model_name, hparams, finetune_only=True, report_to='wandb')
