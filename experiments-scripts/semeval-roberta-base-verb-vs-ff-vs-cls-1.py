from train import repeat_test_model

import logging
logging.getLogger().setLevel(logging.INFO)
handler = logging.StreamHandler()
logging.getLogger().addHandler(handler)

import warnings
warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0")

base_hparams = {
    'masked_lm': True,
    'masked_lm_prompt': 4,
    'clf_hidden_dim': 768,
    'clf_drop_prob': 0.1,
    'ft_epochs': 6,
    'lr_ft': 5e-5,
    'train_ft_batch': 32,
    'ft_eval_steps': 50,
    'warmup_steps_ft': 100,
    'seq_max_len': 85,
}

hparams_list = [
    {'use_semeval_zeroshot': 'AT',
     'masked_lm_verbalizer': True,
     'clf_gelu_head': False,
     'frozen_head': False},

    {'use_semeval_zeroshot': 'AT',
     'masked_lm_verbalizer': False,
     'clf_gelu_head': True,
     'frozen_head': False, },

    {'use_semeval_zeroshot': 'AT',
     'masked_lm': False,
     'use_cls_token': True,
     'clf_gelu_head': True,},

    {'use_semeval_zeroshot': 'DT',
     'masked_lm_verbalizer': True,
     'clf_gelu_head': False,
     'frozen_head': False},

    {'use_semeval_zeroshot': 'DT',
     'masked_lm_verbalizer': False,
     'clf_gelu_head': True,
     'frozen_head': False, },

    {'use_semeval_zeroshot': 'DT',
     'masked_lm': False,
     'use_cls_token': True,
     'clf_gelu_head': True, },

    {'use_semeval_zeroshot': 'HC',
     'masked_lm_verbalizer': True,
     'clf_gelu_head': False,
     'frozen_head': False},

    {'use_semeval_zeroshot': 'HC',
     'masked_lm_verbalizer': False,
     'clf_gelu_head': True,
     'frozen_head': False, },

    {'use_semeval_zeroshot': 'HC',
     'masked_lm': False,
     'use_cls_token': True,
     'clf_gelu_head': True, },

    {'use_semeval_zeroshot': 'CL',
     'masked_lm_verbalizer': True,
     'clf_gelu_head': False,
     'frozen_head': False},

    {'use_semeval_zeroshot': 'CL',
     'masked_lm_verbalizer': False,
     'clf_gelu_head': True,
     'frozen_head': False, },

    {'use_semeval_zeroshot': 'CL',
     'masked_lm': False,
     'use_cls_token': True,
     'clf_gelu_head': True, },

    {'use_semeval_zeroshot': 'AB',
     'masked_lm_verbalizer': True,
     'clf_gelu_head': False,
     'frozen_head': False},

    {'use_semeval_zeroshot': 'AB',
     'masked_lm_verbalizer': False,
     'clf_gelu_head': True,
     'frozen_head': False, },

    {'use_semeval_zeroshot': 'AB',
     'masked_lm': False,
     'use_cls_token': True,
     'clf_gelu_head': True, },

    {'use_semeval_zeroshot': 'FE',
     'masked_lm_verbalizer': True,
     'clf_gelu_head': False,
     'frozen_head': False},

    {'use_semeval_zeroshot': 'FE',
     'masked_lm_verbalizer': False,
     'clf_gelu_head': True,
     'frozen_head': False, },

    {'use_semeval_zeroshot': 'FE',
     'masked_lm': False,
     'use_cls_token': True,
     'clf_gelu_head': True, },
]

for hparams in hparams_list:
    hparams.update(base_hparams)
    model_name = f'roberta-base-verb-vs-ff-vs-cls-1-{hparams["use_semeval_zeroshot"]}'
    repeat_test_model(10, 'roberta-base', model_name, hparams, finetune_only=True, report_to='wandb')
