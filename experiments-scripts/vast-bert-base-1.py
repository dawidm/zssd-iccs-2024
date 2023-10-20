from train import repeat_test_model

import logging
logging.getLogger().setLevel(logging.INFO)
handler = logging.StreamHandler()
logging.getLogger().addHandler(handler)

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

base_hparams = {
    'use_vast': True,
    'use_bert_pooling_out': True,
    'clf_hidden_dim': 768,
    'clf_drop_prob': 0.1,
    'ft_epochs': 6,
    'lr_ft': 5e-5,
    'train_ft_batch': 32,
    'ft_eval_steps': 100,
    'warmup_steps_ft': 100,
    'seq_max_len': 280,
}

hparams_list = [
    {'vast_preprocess_text': True,
     'vast_preprocess_topic': True,
     'vast_filter_ambiguous': False,
     'vast_filter_type_3': False,
     'seq_max_len': 160},
    {'vast_preprocess_text': False,
     'vast_preprocess_topic': True,
     'vast_filter_ambiguous': False,
     'vast_filter_type_3': False},
    {'vast_preprocess_text': False,
     'vast_preprocess_topic': False,
     'vast_filter_ambiguous': False,
     'vast_filter_type_3': False},
]

for hparams in hparams_list:
    hparams.update(base_hparams)
    model_name = f'vast-bert-base-2'
    repeat_test_model(10, 'bert-base-uncased', model_name, hparams, finetune_only=True, report_to='wandb')
