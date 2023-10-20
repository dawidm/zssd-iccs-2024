import datetime
import logging
import math
import os
import subprocess
from pprint import pprint
from typing import Dict

import copy
import numpy as np
import torch
import transformers
from sklearn.metrics import classification_report, f1_score
from transformers import TrainingArguments, Trainer, get_linear_schedule_with_warmup
import wandb

from models import compute_metrics_f1, f1_macro_fa_score, StanceEncoderModel
from stancedatasets import get_datasets

logger = logging.getLogger('stance.train')

def_hparams = {
    # datasets options
    'use_vast': False,
    'use_vast_alt': False,
    'use_semeval': False,
    'use_semeval_zeroshot': False,
    'use_semeval_80_20': False,
    'semeval_zeroshot_eval_on_test': False,
    'semeval_exclude_dt': True,
    'semeval_single_target': None,
    'semeval_adapt_target_names': False,
    'semeval_adapt_target_names_type': 'lowercase',
    'semeval_remove_hashtags': False,
    'semeval_zeroshot_test_only': False,
    'semeval_zeroshot_augment_targets': False,
    'semeval_zs_rename_cl': False,
    'use_covid': False,
    'use_vast_pl': False,
    'use_vast_gen_1': False,
    'use_vast_gen_2': False,
    'use_vast_gen_3': False,
    'use_vast_gen_4': False,
    'use_vast_gpt_1': False,
    'use_seval_gpt_1': False,
    'use_tt_gen_pl_1': False,
    'use_tt_gen_pl_1_short': False,
    'use_clarin_pl': False,
    'use_clarin_zeroshot': None,
    'use_vast_t3': False,
    'gen_neutral_synth_only': False,
    'gen_tt_pl_favor_only': False,
    'gpt_synth_neutral_only': False,
    'vast_preprocess_text': False,
    'vast_preprocess_topic': False,
    'vast_topic_only': False,
    'vast_filter_ambiguous': True,
    'vast_filter_type_3': False,
    'vast_only_true_neutral': False,
    'target_definitions_file': None,
    'concat_augment_train': False,
    'limit_train_samples': None,
    'remove_punctuation_jcl': False, # remove punctuation from JointCL method
    'target_word_limit': None,
    'lowercase_targets': False,

    # model options
    'clf_hidden_dim': 300,
    'clf_drop_prob': 0.2,
    'clf_small_head': False,
    'clf_gelu_head': False,
    'clf_roberta_head': False,
    'use_avg_representations': False,
    'use_bert_pooling_out': False,
    'use_cls_token': False,
    'use_second_cls_token': False,
    'use_2_new_tokens': False,
    'use_maxpool': False,
    'use_topic_pos': False,
    # roberta
    'masked_lm': False,
    'masked_lm_tokens': 1,
    'masked_lm_prompt': 4,
    'masked_lm_new_token': False,
    'return_masked_probas': False,
    'masked_lm_verbalizer': False,
    'masked_lm_like_clf_head': False,

    # training options
    'train_head_batch': 64,
    'eval_head_batch': 128,
    'frozen_head': False,
    'head_epochs': 10,
    'lr_head': 1e-3,
    'weight_decay_head': 0,
    'warmup_steps_head': 0,
    'train_ft_batch': 64,
    'ft_gradient_acc_steps': 1,
    'eval_ft_batch': 128,
    'ft_epochs': 3,
    'lr_ft': 1e-5,
    'weight_decay_ft': 0,
    'lr_ft_classifier': None,
    'weight_decay_ft_classifier': 0,
    'ft_eval_steps': 50,
    'head_eval_steps': 50,
    'warmup_steps_ft': 100,
    'use_best_model': True,
    'seq_max_len': 280,
    'weighted_loss': False,
    'weighted_loss_sqrt': False,
    'class_weights': None,
    'contrastive_loss': False,
    'contrastive_loss_v2': False,
    'contrastive_loss_v3': False,
    'contrastive_temperature': 0.3,
    'contrastive_lambda': 0.9,

    # model selection metrics
    'metric_for_best_model': 'f1_macro',
    'metric_best_greater_is_better': True,

    # topic weighted post representations
    'use_topic_weighted_avg': False,
    'topic_weights_smoothing': None,
}


def _rename_best_checkpoint_dir(trainer):
    # renames best checkpoint dir to have 'best' suffix, eg. checkpoint-100 -> checkpoint-best
    if trainer.state.best_model_checkpoint is not None:
        new_name = trainer.state.best_model_checkpoint.split('-')
        new_name[-1] = 'best'
        new_name = '-'.join(new_name)
        os.rename(trainer.state.best_model_checkpoint, new_name)
        return new_name
    return None


def train_model(base_model_name_or_path, trainer_name, hparams: Dict,
                finetune=True, finetune_only=False, finetune_checkpoint=None,
                seed=1, eval_on_test=True, eval_on_zeroshot_test=True, eval_on_test_no_ft=True,
                eval_on_true_neutral=True,
                vast_eval_on_phenomena=False,
                print_hparams=False, limit_samples=None, report_to='tensorboard',
                save_preds=True):
    """
    model_name: str
        'bert-joint-uncased': BertForJointStance, bert-base-uncased
        'bert-joint-cased': BertForJointStance, bert-base-cased
        'bert-joint-large-uncased': BertForJointStance, bert-large-uncased
        'bert-joint-large-cased': BertForJointStance, bert-large-cased
    """

    print(f'Training started at: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f"git hash: {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}")

    use_wandb = True if report_to == 'wandb' else False

    for current_key in hparams.keys():
        if current_key not in def_hparams.keys():
            raise ValueError(f'wrong hparam: {current_key}')

    new_hparams = copy.deepcopy(def_hparams)
    new_hparams.update(hparams)
    hparams = new_hparams
    if print_hparams:
        pprint(hparams)

    if hparams['use_second_cls_token'] and hparams['use_2_new_tokens']:
        raise ValueError('hparams: use_second_cls_token and use_2_new_tokens cannot be both True')
    if hparams['contrastive_loss'] and hparams['contrastive_loss_v2']:
        raise ValueError('hparams: contrastive_loss and contrastive_loss_v2 cannot be both True')

    if 'herbert' in base_model_name_or_path:
        if hparams['masked_lm']:
            logger.info('using polish prompts for masked LM')
        masked_lm_lang = 'pl'
    else:
        if hparams['masked_lm']:
            logger.info('using english prompts for masked LM')
        masked_lm_lang = 'en'

    transformers.set_seed(seed)

    get_datasets_out = get_datasets(base_model_name_or_path,
                                    vast=hparams['use_vast'],
                                    vast_alt=hparams['use_vast_alt'],
                                    semeval=hparams['use_semeval'],
                                    semeval_zeroshot=hparams['use_semeval_zeroshot'],
                                    semeval_80_20=hparams['use_semeval_80_20'],
                                    covid=hparams['use_covid'],
                                    vast_pl=hparams['use_vast_pl'],
                                    vast_gen_1=hparams['use_vast_gen_1'],
                                    vast_gen_2=hparams['use_vast_gen_2'],
                                    vast_gen_3=hparams['use_vast_gen_3'],
                                    vast_gen_4=hparams['use_vast_gen_4'],
                                    vast_gpt_1=hparams['use_vast_gpt_1'],
                                    seval_gpt_1=hparams['use_seval_gpt_1'],
                                    gen_tt_pl_1=hparams['use_tt_gen_pl_1'],
                                    gen_tt_pl_1_short=hparams['use_tt_gen_pl_1_short'],
                                    clarin_pl=hparams['use_clarin_pl'],
                                    clarin_zeroshot=hparams['use_clarin_zeroshot'],
                                    vast_t3=hparams['use_vast_t3'],
                                    zeroshot_test=eval_on_zeroshot_test,
                                    gen_neutral_synth_only=hparams['gen_neutral_synth_only'],
                                    gen_tt_pl_favor_only=hparams['gen_tt_pl_favor_only'],
                                    gpt_synth_neutral_only=hparams['gpt_synth_neutral_only'],
                                    vast_topic_only=hparams['vast_topic_only'],
                                    limit_samples=limit_samples,
                                    vast_preprocess_text=hparams['vast_preprocess_text'],
                                    vast_preprocess_topic=hparams['vast_preprocess_topic'],
                                    vast_filter_ambiguous=hparams['vast_filter_ambiguous'],
                                    vast_filter_type_3=hparams['vast_filter_type_3'],
                                    vast_only_true_neutral=hparams['vast_only_true_neutral'],
                                    target_definitions_file=hparams['target_definitions_file'],
                                    max_len=hparams['seq_max_len'],
                                    multi_cls_2=hparams['use_second_cls_token'],
                                    two_new_tokens=hparams['use_2_new_tokens'],
                                    masked_lm=hparams['masked_lm'],
                                    masked_lm_tokens=hparams['masked_lm_tokens'],
                                    masked_lm_prompt=hparams['masked_lm_prompt'],
                                    masked_lm_lang=masked_lm_lang,
                                    masked_lm_new_token=hparams['masked_lm_new_token'],
                                    concat_augment_train=hparams['concat_augment_train'],
                                    semeval_zeroshot_eval_on_test=hparams['semeval_zeroshot_eval_on_test'],
                                    limit_train_samples=hparams['limit_train_samples'],
                                    semeval_exclude_dt=hparams['semeval_exclude_dt'],
                                    semeval_single_target=hparams['semeval_single_target'],
                                    semeval_adapt_target_names=hparams['semeval_adapt_target_names'],
                                    semeval_adapt_target_names_type=hparams['semeval_adapt_target_names_type'],
                                    semeval_remove_hashtags=hparams['semeval_remove_hashtags'],
                                    semeval_zeroshot_test_only=hparams['semeval_zeroshot_test_only'],
                                    semeval_zeroshot_augment_targets=hparams['semeval_zeroshot_augment_targets'],
                                    semeval_zs_rename_cl=hparams['semeval_zs_rename_cl'],
                                    remove_punctuation_jcl=hparams['remove_punctuation_jcl'],
                                    target_word_limit=hparams['target_word_limit'],
                                    lowercase_targets=hparams['lowercase_targets'],)

    train_dataset, dev_dataset, test_dataset = get_datasets_out['datasets']
    train_df, dev_df, test_df = get_datasets_out['dfs']
    zeroshot_test_dataset = get_datasets_out.get('zeroshot_test_dataset', None)
    true_neutral_test_dataset = get_datasets_out.get('true_neutral_test_dataset', None)
    special_tokens = get_datasets_out.get('special_tokens', None)
    if hparams['use_2_new_tokens'] and special_tokens is None:
        raise ValueError('missing special_tokens for classification (use_2_new_tokens)')
    mask_token_id = get_datasets_out['mask_token_id']
    verbalizer_token_ids = get_datasets_out['verbalizer_token_ids']

    task_specific_params = {'num_labels': 3,
                            'mask_token_id': mask_token_id,
                            'verbalizer_token_ids': verbalizer_token_ids,
                            'use_avg_representations': hparams['use_avg_representations'],
                            'use_cls_token': hparams['use_cls_token'],
                            'use_bert_pooling_out': hparams['use_bert_pooling_out'],
                            'use_maxpool': hparams['use_maxpool'],
                            'clf_hidden_dim': hparams['clf_hidden_dim'],
                            'clf_drop_prob': hparams['clf_drop_prob'],
                            'clf_small_head': hparams['clf_small_head'],
                            'clf_gelu_head': hparams['clf_gelu_head'],
                            'clf_roberta_head': hparams['clf_roberta_head'],
                            'use_topic_pos': hparams['use_topic_pos'],
                            'use_topic_weighted_avg': hparams['use_topic_weighted_avg'],
                            'topic_weights_smoothing': hparams['topic_weights_smoothing'],
                            'use_second_cls_token': hparams['use_second_cls_token'],
                            'custom_cls_tokens': special_tokens if hparams['use_2_new_tokens'] else None,
                            'masked_lm_new_token': special_tokens[0] if hparams['masked_lm_new_token'] else None,
                            'masked_lm': hparams['masked_lm'],
                            'masked_lm_n_tokens': hparams['masked_lm_tokens'],
                            'masked_lm_verbalizer': hparams['masked_lm_verbalizer'],
                            'masked_lm_like_clf_head': hparams['masked_lm_like_clf_head'],
                            'return_masked_probas': hparams['return_masked_probas'],
                            'weighted_loss': hparams['weighted_loss'],
                            'contrastive_loss': hparams['contrastive_loss'],
                            'contrastive_loss_v2': hparams['contrastive_loss_v2'],
                            'contrastive_loss_v3': hparams['contrastive_loss_v3'],
                            'contrastive_temperature': hparams['contrastive_temperature'],
                            'contrastive_lambda': hparams['contrastive_lambda'],
                            'class_weights': hparams['class_weights'],
                            'weighted_loss_sqrt': hparams['weighted_loss_sqrt'],}

    model = StanceEncoderModel.instantiate_from_base(base_model_name_or_path,
                                                     task_specific_params)

    model.base_enc_model.resize_token_embeddings(model.config.vocab_size + (len(special_tokens) if special_tokens else 0))

    if 0 < hparams['warmup_steps_ft'] < 1:
        total_steps = hparams['ft_epochs'] * math.ceil(len(train_dataset)/hparams['train_ft_batch'])
        hparams['warmup_steps_ft'] = int(hparams['warmup_steps_ft'] * total_steps)
        logger.info(f"warmup steps: {hparams['warmup_steps_ft']}")

    if hparams['ft_eval_steps'] == -1:
        hparams['ft_eval_steps'] = math.ceil(len(train_dataset)/hparams['train_ft_batch'])
        logger.info(f"ft eval steps: {hparams['ft_eval_steps']} (1 epoch)")
    if hparams['head_eval_steps'] == -1:
        hparams['head_eval_steps'] = math.ceil(len(train_dataset)/hparams['train_head_batch'])
        logger.info(f"head eval steps: {hparams['head_eval_steps']} (1 epoch)")

    results = {}

    device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1

    if not finetune_only:
        model.freeze_pretrained()

        training_args = TrainingArguments(trainer_name,
                                          run_name=trainer_name,
                                          per_device_train_batch_size=hparams['train_head_batch']//device_count,
                                          per_device_eval_batch_size=hparams['eval_head_batch']//device_count,
                                          evaluation_strategy='steps',
                                          save_strategy='steps',
                                          save_total_limit=1,
                                          save_steps=hparams['head_eval_steps'],
                                          eval_steps=hparams['head_eval_steps'],
                                          logging_steps=hparams['head_eval_steps'],
                                          num_train_epochs=hparams['head_epochs'],
                                          learning_rate=hparams['lr_head'],
                                          weight_decay=hparams['weight_decay_head'],
                                          warmup_steps=hparams['warmup_steps_head'],
                                          load_best_model_at_end=hparams['use_best_model'],
                                          metric_for_best_model=hparams['metric_for_best_model'],
                                          greater_is_better=hparams['metric_best_greater_is_better'],
                                          report_to=report_to,
                                          seed=seed
                                          )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics = compute_metrics_f1,
        )

        if use_wandb:
            wandb.init(project='stance', name=trainer_name, config=hparams)

        trainer.train()
        checkpoint_dir = _rename_best_checkpoint_dir(trainer)
        if checkpoint_dir:
            get_datasets_out['tokenizer'].save_pretrained(checkpoint_dir)

        if eval_on_test_no_ft:
            preds = trainer.predict(test_dataset)
            results['test_f1_no_ft'] = f1_score(test_dataset.labels, preds.predictions.argmax(axis=1), average='macro')
            results['test_f1_fa_no_ft'] = f1_macro_fa_score(test_dataset.labels, preds.predictions.argmax(axis=1))
            print(classification_report(test_dataset.labels, preds.predictions.argmax(axis=1), digits=3))
            print('f1 macro fa: ', results['test_f1_fa_no_ft'])

        if use_wandb:
            wandb.finish()

    if finetune or finetune_only:

        training_args = TrainingArguments(f'{trainer_name}-ft',
                                          run_name=f'{trainer_name}-ft',
                                          per_device_train_batch_size=hparams['train_ft_batch']//device_count,
                                          per_device_eval_batch_size=hparams['eval_ft_batch']//device_count,
                                          gradient_accumulation_steps=hparams['ft_gradient_acc_steps'],
                                          evaluation_strategy='steps',
                                          save_strategy='steps',
                                          save_total_limit=1,
                                          save_steps=hparams['ft_eval_steps'],
                                          eval_steps=hparams['ft_eval_steps'],
                                          logging_steps=hparams['ft_eval_steps'],
                                          num_train_epochs=hparams['ft_epochs'],
                                          learning_rate=hparams['lr_ft'],
                                          warmup_steps=hparams['warmup_steps_ft'],
                                          load_best_model_at_end=hparams['use_best_model'],
                                          metric_for_best_model=hparams['metric_for_best_model'],
                                          greater_is_better=hparams['metric_best_greater_is_better'],
                                          report_to=report_to,
                                          seed=seed
                                          )

        if hparams['lr_ft_classifier'] is not None:
            optim_params = [{'params': model.base_enc_model.parameters()},
                            {'params': model.classifier.parameters(),
                             'lr': hparams['lr_ft_classifier'],
                             'weight_decay': hparams['weight_decay_ft_classifier']}, ]
        else:
            params = model.base_enc_model.parameters() if hparams['frozen_head'] else model.parameters()
            optim_params = [{'params': params}]
        optimizer = torch.optim.AdamW(optim_params, lr=hparams['lr_ft'], weight_decay=hparams['weight_decay_ft'])
        training_steps = len(train_dataset) // hparams['train_ft_batch'] * hparams['ft_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=hparams['warmup_steps_ft'],
                                                    num_training_steps=training_steps)

        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics_f1,
            optimizers=(optimizer, scheduler)
        )

        if finetune_checkpoint is not None:
            checkpoint = torch.load(os.path.join(finetune_checkpoint, 'pytorch_model.bin'))
            trainer.model.load_state_dict(checkpoint)

        model.unfreeze_pretrained()

        if use_wandb:
            wandb.init(project='stance', name=trainer_name + '-ft', config=hparams)

        trainer.train()

        checkpoint_dir = _rename_best_checkpoint_dir(trainer)
        if checkpoint_dir:
            get_datasets_out['tokenizer'].save_pretrained(checkpoint_dir)

    if eval_on_test or save_preds:
        preds = trainer.predict(test_dataset)
        test_df['pred'] = preds.predictions.argmax(axis=1)
        if hparams['use_semeval_zeroshot']:
            preds_file_name = f'{base_model_name_or_path}_{hparams["use_semeval_zeroshot"]}_test_preds.pkl'
        else:
            preds_file_name = f'{base_model_name_or_path}_test_preds.pkl'
        test_df.to_pickle(preds_file_name)

    if eval_on_test:
        results['test_f1'] = f1_score(test_dataset.labels, preds.predictions.argmax(axis=1), average='macro')
        results['test_f1_fa'] = f1_macro_fa_score(test_dataset.labels, preds.predictions.argmax(axis=1))
        print('test results: ')
        print(classification_report(test_dataset.labels, preds.predictions.argmax(axis=1), digits=3))
        print('f1 macro fa: ', results['test_f1_fa'])
        if vast_eval_on_phenomena:
            test_df['preds'] = preds.predictions.argmax(axis=1)
            test_df['true'] = test_dataset.labels
            for phenomenon in ['indirect_stance', 'doc_multiple_labels', 'quotations', 'sarcasm', 'doc_multiple_topics']:
                print(phenomenon)
                test_df_phenomenon = test_df[test_df[phenomenon] == 1]
                print(classification_report(test_df_phenomenon['true'], test_df_phenomenon['preds'], digits=3))

    if eval_on_zeroshot_test:
        if hparams['use_vast'] is True or (type(hparams['use_vast']) is list and 'test' in hparams['use_vast']):
            preds = trainer.predict(zeroshot_test_dataset)
            results['zeroshot_test_f1'] = f1_score(zeroshot_test_dataset.labels, preds.predictions.argmax(axis=1), average='macro')
            results['zeroshot_test_f1_fa'] = f1_macro_fa_score(zeroshot_test_dataset.labels, preds.predictions.argmax(axis=1))
            print('zero-shot test results: ')
            print(classification_report(zeroshot_test_dataset.labels, preds.predictions.argmax(axis=1), digits=3))
            print('f1 macro fa: ', results['zeroshot_test_f1_fa'])

    if eval_on_true_neutral:
        if hparams['use_vast'] is True or (type(hparams['use_vast']) is list and 'test' in hparams['use_vast']):
            preds = trainer.predict(true_neutral_test_dataset)
            results['true_neutral_test_f1'] = f1_score(true_neutral_test_dataset.labels, preds.predictions.argmax(axis=1), average='macro')
            results['true_neutral_test_f1_fa'] = f1_macro_fa_score(true_neutral_test_dataset.labels, preds.predictions.argmax(axis=1))
            print('true neutral test results: ')
            print(classification_report(true_neutral_test_dataset.labels, preds.predictions.argmax(axis=1), digits=3))
            print('f1 macro fa: ', results['true_neutral_test_f1_fa'])


    eval_f1_macros = []
    for log_item in trainer.state.log_history:
        if 'eval_f1_macro' in log_item:
            eval_f1_macros.append(log_item['eval_f1_macro'])
    results['dev_f1'] = np.max(eval_f1_macros)

    results['dev_f1_mean_3_best'] = np.sort(eval_f1_macros)[-3:].mean()

    wandb.log(results)
    if use_wandb:
        wandb.finish()

    return trainer, results


def repeat_test_model(n_reps, model_name, trainer_name, hparams, finetune_only=False, finetune=True, limit_samples=None,
                      semeval_zs_targets=None, semeval_zs_seed=0, report_to='tensorboard', seed_add=0):
    if report_to == 'wandb':
        try:
            wandb.finish(quiet=True)
        except:
            pass

    dev_f1s = []
    test_f1s = []
    test_f1s_fa = []
    test_zs_f1s = []

    if semeval_zs_targets:
        print('semeval zs targets provided, training one model for each target')
        n_reps = len(semeval_zs_targets)

    for i in range(n_reps):
        print(f'target: {semeval_zs_targets[i]}' if semeval_zs_targets else f'run {i}')
        if semeval_zs_targets:
            hparams['use_semeval_zeroshot'] = semeval_zs_targets[i]
        seed = semeval_zs_seed if semeval_zs_targets else i
        seed += seed_add
        results = train_model(base_model_name_or_path=model_name, trainer_name=trainer_name, hparams=hparams, seed=seed,
                              finetune_only=finetune_only, finetune=finetune, limit_samples=limit_samples,
                              report_to=report_to)[1]
        dev_f1s.append(results['dev_f1'])
        test_f1s.append(results['test_f1'])
        test_f1s_fa.append(results['test_f1_fa'])
        if 'zeroshot_test_f1' in results:
            test_zs_f1s.append(results['zeroshot_test_f1'])

    print(f'dev f1s: mean: {np.mean(dev_f1s)} std: {np.std(dev_f1s)}')
    print(f'test f1s: mean: {np.mean(test_f1s)} std: {np.std(test_f1s)}')
    print(f'test f1s fa: mean: {np.mean(test_f1s_fa)} std: {np.std(test_f1s_fa)}')
    print(f'test zeroshot f1s: mean: {np.mean(test_zs_f1s)} std: {np.std(test_zs_f1s)}')
