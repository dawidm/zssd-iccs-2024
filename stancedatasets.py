import logging
import random
import re

import numpy as np
import pandas as pd
import torch
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from processing_utils import _remove_punctuation_jcl

logger = logging.getLogger('stancedatasets')

# targets names
CLASS_NAMES = ['favor', 'against', 'neither']

# default verbalizer (classification with prompt) words for classes
DEFAULT_VERBALIZERS = ['positive', 'negative', 'neutral']

SE_SHORT_TARGETS = {
    'HC': 'hillary clinton',
    'DT': 'donald trump',
    'AB': 'legalization of abortion',
    'CL': 'climate change is a real concern',
    'FE': 'feminist movement',
    'AT': 'atheism',
}

_SEQ2SEQ_TOKENIZERS = {'t5-base', 't5-small'}

PROMPT_TYPE_TARGET_ONLY = 10000  # mlm prompt that contains only target and mask/new token before it


def load_gpt_dataset(file_prefix,
                     add_stance_class=True,
                     synth_neutral_only=False,
                     multi_only=True,):
    df_train = pd.read_csv('data/' + file_prefix + '_train.csv')
    df_dev = pd.read_csv('data/' + file_prefix + '_dev.csv')
    if synth_neutral_only:
        df_train = df_train[(df_train['is_synth_neutral'].astype(bool)) | (df_train['stance'] != 'neither')]
        df_dev = df_dev[(df_dev['is_synth_neutral'].astype(bool)) | (df_dev['stance'] != 'neither')]
    if multi_only:
        df_train = df_train[df_train['multi_target'].astype(bool) | (df_train['stance'] == 'neither')]
        df_dev = df_dev[df_dev['multi_target'].astype(bool) | (df_dev['stance'] == 'neither')]
    if add_stance_class:
        _add_stance_class(df_train)
        _add_stance_class(df_dev)
    return df_train, df_dev


def load_gen_tt_pl_1_dataset(train_path='data/gen_tt_pl_1.csv',
                             add_stance_class=True,
                             random_state=1,
                             fovor_only=False,
                             short_target_only=False,
                             short_target_n_tokens=5):
    df = pd.read_csv(train_path)
    if fovor_only:
        df = df[df['stance'] == 'favor']
    if add_stance_class:
        _add_stance_class(df)
    if short_target_only:
        df = df[df['target'].apply(lambda x: len(x.split())) <= short_target_n_tokens]
    df_train, df_test = train_test_split(df, train_size=0.9, random_state=random_state)
    return df_train, df_test


def load_clarin_dataset(train_path='data/stance_clarin.csv',
                        add_stance_class=True, dev_size=0.15, seed=1):
    df = pd.read_csv(train_path)
    train_df, dev_df = train_test_split(df, test_size=dev_size, random_state=seed)
    if add_stance_class:
        _add_stance_class(train_df)
        _add_stance_class(dev_df)
    return train_df, dev_df


def load_clarin_zs_dataset(zs_target, train_path='data/stance_clarin.csv',
                           add_stance_class=True, dev_size=0.15, seed=1):
    df = pd.read_csv(train_path)
    test_df = df[df['target'] == zs_target]
    df = df[df['target'] != zs_target]
    train_df, dev_df = train_test_split(df, test_size=dev_size, random_state=seed)
    if add_stance_class:
        _add_stance_class(train_df)
        _add_stance_class(dev_df)
        _add_stance_class(test_df)
    return train_df, dev_df, test_df


def load_gen_vast_1_dataset(train_path='data/gen_vast_1.csv',
                            add_stance_class=True,
                            gen_neutral_synth_only=False):
    df = pd.read_csv(train_path)
    if gen_neutral_synth_only:
        df = df[~((df['stance'] == 'neither') & (df['is_synthetic_neutral'] == False))]
    if add_stance_class:
        _add_stance_class(df)
    return df


def load_gen_vast_2_dataset(add_stance_class=True,
                            gen_neutral_synth_only=False):
    return load_gen_vast_1_dataset(train_path='data/gen_vast_2.csv',
                                   add_stance_class=add_stance_class,
                                   gen_neutral_synth_only=gen_neutral_synth_only)


def load_gen_vast_3_dataset(add_stance_class=True,
                            gen_neutral_synth_only=False):
    return load_gen_vast_1_dataset(train_path='data/gen_vast_3.csv',
                                   add_stance_class=add_stance_class,
                                   gen_neutral_synth_only=gen_neutral_synth_only)


def load_gen_vast_4_dataset(add_stance_class=True,
                            gen_neutral_synth_only=False):
    return load_gen_vast_1_dataset(train_path='data/gen_vast_4.csv',
                                   add_stance_class=add_stance_class,
                                   gen_neutral_synth_only=gen_neutral_synth_only)


def load_gen_vast_s_1_dataset(add_stance_class=True,
                              gen_neutral_synth_only=False):
    return load_gen_vast_1_dataset(train_path='data/gen_vast_s_1.csv',
                                   add_stance_class=add_stance_class,
                                   gen_neutral_synth_only=gen_neutral_synth_only)


def load_semeval_dataset(train_path='data/se_train.csv',
                         test_path='data/se_test.csv',
                         add_stance_class=True,
                         adapt_target_names=True,
                         adapt_target_names_type='lowercase',
                         exclude_dt=True,
                         random_val_dataset=True,
                         train_proportion=0.9,
                         single_target=None,
                         train_val_80_20=False,
                         remove_hashtags=False,
                         seed=1, ):
    se_2016_A_train_df = adapt_semeval_df(pd.read_csv(train_path), remove_hashtags=remove_hashtags)
    se_2016_A_test_df = adapt_semeval_df(pd.read_csv(test_path), remove_hashtags=remove_hashtags)
    if train_val_80_20:
        cat_df = pd.concat([se_2016_A_train_df, se_2016_A_test_df])
        se_2016_A_train_df, se_2016_A_test_df = train_test_split(cat_df, test_size=0.2, random_state=seed)
    if exclude_dt:
        se_2016_A_test_df = se_2016_A_test_df[se_2016_A_test_df['target'] != 'Donald Trump']
    else:
        se_2016_A_test_no_dt = se_2016_A_test_df[se_2016_A_test_df['target'] != 'Donald Trump']
        se_2016_A_test_dt = se_2016_A_test_df[se_2016_A_test_df['target'] == 'Donald Trump']
        test_to_all_proportion = len(se_2016_A_test_no_dt) / (len(se_2016_A_test_no_dt) + len(se_2016_A_train_df))
        dt_train, dt_test = train_test_split(se_2016_A_test_dt, test_size=test_to_all_proportion, random_state=seed)
        se_2016_A_train_df = pd.concat([se_2016_A_train_df, dt_train])
        se_2016_A_test_df = pd.concat([se_2016_A_test_no_dt, dt_test])
    if add_stance_class:
        _add_stance_class(se_2016_A_train_df)
        _add_stance_class(se_2016_A_test_df)
    if adapt_target_names:
        if adapt_target_names_type == 'lowercase':
            adapt_f = lambda x: x.lower()
        elif adapt_target_names_type == 'lower-exclude-names':
            adapt_f = lambda x: x.lower() if x.lower() not in ['hillary clinton', 'donald trump'] else x
        else:
            raise ValueError('unknown adapt_target_names_type: {}'.format(adapt_target_names_type))
        se_2016_A_train_df['target'] = se_2016_A_train_df['target'].apply(adapt_f)
        se_2016_A_test_df['target'] = se_2016_A_test_df['target'].apply(adapt_f)
    if single_target is not None:
        se_2016_A_train_df = se_2016_A_train_df[
            se_2016_A_train_df['target'].str.lower() == SE_SHORT_TARGETS[single_target]]
        se_2016_A_test_df = se_2016_A_test_df[
            se_2016_A_test_df['target'].str.lower() == SE_SHORT_TARGETS[single_target]]
    if random_val_dataset:
        se_2016_A_train_df, se_2016_A_dev_df = \
            train_test_split(se_2016_A_train_df, train_size=train_proportion, random_state=seed,
                             stratify=list(se_2016_A_train_df['target'] + se_2016_A_train_df['stance']))
        return se_2016_A_train_df, se_2016_A_dev_df, se_2016_A_test_df
    else:
        return se_2016_A_train_df, se_2016_A_test_df, se_2016_A_test_df


def load_semeval_zeroshot_dataset(zeroshot_target: str,
                                  train_path='data/se_train.csv',
                                  dev_path='data/se_test.csv',
                                  add_stance_class=True,
                                  adapt_target_names=True,
                                  adapt_target_names_type='lowercase',
                                  train_proportion=0.9,
                                  eval_on_test=False,
                                  remove_hashtags=False,
                                  augment_targets=False,
                                  rename_cl=False,
                                  seed=1, ):
    if zeroshot_target not in SE_SHORT_TARGETS.keys():
        raise ValueError(f"Invalid zeroshot target: {zeroshot_target}. "
                         f"Valid values: {SE_SHORT_TARGETS}")
    train_df, dev_df, _ = load_semeval_dataset(train_path, dev_path, add_stance_class,
                                               adapt_target_names, adapt_target_names_type,
                                               exclude_dt=False, random_val_dataset=False,
                                               remove_hashtags=remove_hashtags, seed=seed)
    all_df = pd.concat((train_df, dev_df), axis=0, ignore_index=True)
    train_df = all_df[all_df['target'].str.lower() != SE_SHORT_TARGETS[zeroshot_target]]
    if eval_on_test:
        dev_df = test_df = all_df[all_df['target'].str.lower() == SE_SHORT_TARGETS[zeroshot_target]]
    else:
        train_df, dev_df = train_test_split(train_df, train_size=train_proportion, random_state=seed,
                                            stratify=list(train_df['target'] + train_df['stance']))
        test_df = all_df[all_df['target'].str.lower() == SE_SHORT_TARGETS[zeroshot_target]]

    if augment_targets:
        def switch_fav_against(label):
            if label == 0:
                return 1
            if label == 1:
                return 0
            return label
        aug_dt = train_df[train_df['target'].str.lower() == 'donald trump']
        aug_dt_good = aug_dt.copy()
        aug_dt_good['target'] = 'Donald Trump is a Good Candidate'
        aug_dt_bad = aug_dt.copy()
        aug_dt_bad['target'] = 'Donald Trump is a Bad Candidate'
        aug_dt_bad['class'] = aug_dt_bad['class'].apply(switch_fav_against)
        aug_hc = train_df[train_df['target'].str.lower() == 'hillary clinton']
        aug_hc_good = aug_hc.copy()
        aug_hc_good['target'] = 'Hillary Clinton is a Good Candidate'
        aug_hc_bad = aug_hc.copy()
        aug_hc_bad['target'] = 'Hillary Clinton is a Bad Candidate'
        aug_hc_bad['class'] = aug_hc_bad['class'].apply(switch_fav_against)
        train_df = pd.concat((train_df, aug_dt_good, aug_dt_bad, aug_hc_good, aug_hc_bad), axis=0, ignore_index=True)
    if rename_cl and zeroshot_target == 'CL':
        test_df['target'] = 'Climate Change Awareness'

    return train_df, dev_df, test_df


def load_covid_dataset(train_path='data/covid_train.csv',
                       dev_path='data/covid_dev.csv',
                       test_path='data/covid_test.csv',
                       add_stance_class=True,
                       adapt_target_names=True):
    def _adapt_target_name(target):
        if target == 'fauci':
            return 'Fauci'
        if target == 'face_masks':
            return 'face masks'
        if target == 'stay_at_home_orders':
            return 'stay at home orders'
        if target == 'school_closures':
            return 'school closures'

    def _adapt_target_names(targets: pd.Series):
        return targets.apply(_adapt_target_name)

    df_train = adapt_covid_df(
        pd.read_csv(train_path,
                    index_col=[0]))
    df_dev = adapt_covid_df(
        pd.read_csv(dev_path, index_col=[0]))
    df_test = adapt_covid_df(
        pd.read_csv(test_path,
                    index_col=[0]))
    if add_stance_class:
        _add_stance_class(df_train)
        _add_stance_class(df_dev)
        _add_stance_class(df_test)

    if adapt_target_names:
        df_train['target'] = _adapt_target_names(df_train['target'])
        df_dev['target'] = _adapt_target_names(df_dev['target'])
        df_test['target'] = _adapt_target_names(df_test['target'])

    return df_train, df_dev, df_test


def load_vast_dataset(train_path='data/vast_train_corr.csv',
                      dev_path='data/vast_dev_corr.csv',
                      test_path='data/vast_test_corr.csv',
                      adapt=True,
                      use_preprocessed_post=False,
                      use_preprocessed_topic=False,
                      add_stance_class=True,
                      contains_topic_only=False,
                      polish=False,
                      filter_ambiguous=False,
                      filter_type_3=False,
                      only_true_neutral=False):
    if adapt == False:
        add_stance_class = False
    vast_train = pd.read_csv(train_path)
    vast_dev = pd.read_csv(dev_path)
    vast_test = pd.read_csv(test_path)
    if contains_topic_only:
        vast_train = vast_train[(vast_train['contains_topic?'] == 1) | (vast_train['label'] == 2)]
        vast_dev = vast_dev[(vast_dev['contains_topic?'] == 1) | (vast_dev['label'] == 2)]
        vast_test = vast_test[(vast_test['contains_topic?'] == 1) | (vast_test['label'] == 2)]
    vast_test['zero_shot'] = ~(vast_test['seen?'].astype(bool))
    if only_true_neutral:
        vast_train = vast_train[vast_train['type_idx'] != 4]
        vast_dev = vast_dev[vast_dev['type_idx'] != 4]
        vast_test = vast_test[vast_test['type_idx'] != 4]
    if adapt:
        vast_train = adapt_vast_df(vast_train, use_preprocessed_post=use_preprocessed_post,
                                   use_preprocessed_topic=use_preprocessed_topic, polish=polish,
                                   filter_ambiguous=filter_ambiguous, filter_type_3=filter_type_3)
        vast_dev = adapt_vast_df(vast_dev, use_preprocessed_post=use_preprocessed_post,
                                 use_preprocessed_topic=use_preprocessed_topic, polish=polish,
                                 filter_ambiguous=False, filter_type_3=False)
        vast_test = adapt_vast_df(vast_test, use_preprocessed_post=use_preprocessed_post,
                                  use_preprocessed_topic=use_preprocessed_topic, polish=polish,
                                  filter_ambiguous=False, filter_type_3=False)
    if add_stance_class:
        _add_stance_class(vast_train)
        _add_stance_class(vast_dev)
        _add_stance_class(vast_test)
    return vast_train, vast_dev, vast_test


def load_vast_t3_dataset(train_path='data/vast_train_type_3_gpt.csv'):
    df = pd.read_csv(train_path)
    df = df.rename(columns={'gpt_pred': 'class'})
    return df


def add_target_definitions_to_df(df, defs_csv_file):
    init_targets = set(df['target'])
    defs_df = pd.read_csv(defs_csv_file)
    df = df.merge(defs_df, on='target')
    missing_list = list(init_targets - set(defs_df['target']))
    if len(missing_list) > 0:
        logger.warning(f"{len(missing_list)} targets were not found in the target "
                       f"definitions file: {', '.join(missing_list[0:3])}" +
                       ("..." if len(missing_list) > 3 else ""))
    df['definition'].fillna(df['target'], inplace=True)
    return df


# 1 - target - The tweet explicitly expresses opinion about the target, a part of the target, or an aspect of the target.
# 2 - other - The tweet does NOT expresses opinion about the target but it HAS opinion about something or someone other than the target.
# 3 - none - The tweet is not explicitly expressing opinion. (For example, the tweet is simply giving information.)
def semeval_convert_opinion_towards(opinion_towards: str):
    if opinion_towards.startswith('1.'):
        return 'target'
    if opinion_towards.startswith('2.'):
        return 'other'
    if opinion_towards.startswith('3.'):
        return 'none'
    raise ValueError('wrong value')


def vast_convert_stance(stance: int):
    if stance == 0:
        return 'against'
    if stance == 1:
        return 'favor'
    if stance == 2:
        return 'neither'
    raise ValueError('wrong value')


def adapt_general_df(df: pd.DataFrame):
    df['text_len'] = df['text'].str.len()


def adapt_semeval_df(df: pd.DataFrame, remove_hashtags=False):
    df = df.rename(
        columns={'Tweet': 'text', 'Target': 'target', 'Stance': 'stance', 'Opinion Towards': 'opinion_towards',
                 'Sentiment': 'sentiment'})
    if remove_hashtags:
        def remove_hashtags(texts):
            for i, text in enumerate(texts):
                text_split = text.split()
                while text_split[-1].startswith('#'):
                    text_split.pop()
                texts[i] = ' '.join(text_split)

        remove_hashtags(df['text'])
    adapt_general_df(df)
    df['opinion_towards'] = df['opinion_towards'].apply(semeval_convert_opinion_towards)
    df['stance'] = df['stance'].str.lower()
    df['text'] = df['text'].str.replace('#SemST', '')
    return df


def adapt_covid_df(df: pd.DataFrame):
    df = df.rename(columns={'Tweet Id': 'tweet_id', 'Tweet': 'text', 'Target': 'target', 'Stance': 'stance',
                            'Opinion Towards': 'opinion_towards', 'Sentiment': 'sentiment'})
    adapt_general_df(df)
    df = df[~df['text'].isna()]
    df['opinion_towards'] = df['opinion_towards'].apply(semeval_convert_opinion_towards)  # same as semeval
    df['stance'] = df['stance'].str.lower()
    return df


def _vast_filter_ambiguous(vast_df, preprocessed_post=False, preprocessed_topic=False):
    post_col = 'post' if not preprocessed_post else 'text_s'
    topic_col = 'new_topic_corr' if not preprocessed_topic else 'topic_str'
    test_df_grouped = vast_df.groupby([post_col, topic_col]).nunique().reset_index()
    invalid_samples = test_df_grouped[test_df_grouped['label'] > 1]
    to_remove = set(invalid_samples[post_col] + invalid_samples[topic_col])
    return vast_df[~(vast_df[post_col] + vast_df[topic_col]).isin(to_remove)]


def adapt_vast_df(df: pd.DataFrame, use_preprocessed_post=False, use_preprocessed_topic=False,
                  polish=False, filter_ambiguous=False, filter_type_3=False):
    if polish and (use_preprocessed_topic or use_preprocessed_topic):
        raise ValueError('no preprocessed values for polish dataset')
    if filter_type_3:
        df = df[(df['type_idx'] != 3)]
    if filter_ambiguous:
        old_len = len(df)
        df = _vast_filter_ambiguous(df, preprocessed_post=use_preprocessed_post,
                                    preprocessed_topic=use_preprocessed_topic)
        logger.info(
            f'vast: filtered out {old_len - len(df)} ambiguous samples (old n samples: {old_len}, new n samples: {len(df)})')
    df = df.drop(columns=['text'])
    if use_preprocessed_post:
        text_col = 'text_s'
    elif polish:
        text_col = 'post_pl'
    else:
        text_col = 'post'
    if use_preprocessed_topic:
        topic_col = 'topic_str'
    elif polish:
        topic_col = 'new_topic_pl'
    else:
        topic_col = 'new_topic_corr'
    df = df.rename(columns={text_col: 'text', topic_col: 'target', 'label': 'stance',
                            'seen?': 'seen', 'contains_topic?': 'contains_topic', })
    df['opinion_towards'] = None
    adapt_general_df(df)
    df['stance'] = df['stance'].apply(vast_convert_stance)
    return_cols = ['text', 'target', 'stance', 'text_len', 'seen', 'contains_topic', 'type_idx']
    if set(df.columns).issuperset({'Imp', 'mlS', 'Qte', 'Sarc', 'mlT'}):
        df = df.rename(columns={'Imp': 'indirect_stance', 'mlS': 'doc_multiple_labels', 'Qte': 'quotations',
                                'Sarc': 'sarcasm', 'mlT': 'doc_multiple_topics'})
        return_cols += ['indirect_stance', 'doc_multiple_labels', 'quotations', 'sarcasm', 'doc_multiple_topics']

    return df[return_cols]


def _add_stance_class(df):
    def map_class(x):
        if x == 'favor':
            return 0
        if x == 'against':
            return 1
        if x == 'none' or x == 'neither':
            return 2

    df['class'] = df['stance'].apply(map_class).astype(int)


class StanceDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def format_masked_lm_prompt(stance_df, masked_lm_tokens, tokenizer, prompt_type, lang='en', new_token=None):
    # tokenizer.additional_special_tokens
    if new_token is not None:
        mask_token = new_token
    else:
        mask_token = tokenizer.mask_token
    masks_str = "".join([mask_token] * masked_lm_tokens)
    if prompt_type == PROMPT_TYPE_TARGET_ONLY:
        return list(stance_df['text']), \
            list(masks_str + ' ' + stance_df['target'])
    if lang == 'en':
        if prompt_type == 1:
            return list(stance_df['text']), \
                list('My stance towards ' + stance_df['target'] + ' is: ' + masks_str + '.')
        elif prompt_type == 2:
            return list(stance_df['text']), \
                list('My stance towards ' + stance_df['target'] + ' is ' + masks_str + '.')
        elif prompt_type == 3:
            return list(stance_df['text']), \
                list('Therefore my stance towards ' + stance_df['target'] + ' is: ' + masks_str + '.')
        elif prompt_type == 4:
            return list(stance_df['text']), \
                list('Therefore my stance towards ' + stance_df['target'] + ' is ' + masks_str + '.')
        elif prompt_type == 5:
            return list(stance_df['text']), \
                list('Therefore my stance towards \"' + stance_df['target'] + '\" is ' + masks_str + '.')
        elif prompt_type == 10:
            return list(stance_df['text']), \
                list('The stance towards ' + stance_df['target'] + ' is ' + masks_str + '.')
        elif prompt_type == 100:
            return list(stance_df['text']), \
                list(stance_df['target'] + tokenizer.sep_token + masks_str)
        elif prompt_type == 101:
            return list(stance_df['text']), \
                list(stance_df['target'] + tokenizer.sep_token + 'My stance is ' + masks_str + '.')
        elif prompt_type == 200:
            return list(stance_df['text']), \
                list(stance_df['target'] + '? ' + masks_str + '.')
        elif prompt_type == 1000:
            l_1 = list('My stance towards ' + stance_df['target'] + ' is ' + masks_str + '.')
            l_2 = list('Therefore my stance towards ' + stance_df['target'] + ' is ' + masks_str + '.')
            l_3 = list('The stance towards ' + stance_df['target'] + ' is ' + masks_str + '.')
            mixed_list = []
            for item1, item2, item3 in zip(l_1, l_2, l_3):
                selected_item = random.choice([item1, item2, item3])
                mixed_list.append(selected_item)
            return list(stance_df['text']), mixed_list
        elif prompt_type == 1001:
            l_1 = list('My stance towards ' + stance_df['target'] + ' is ' + masks_str + '.')
            l_2 = list('My feeling about ' + stance_df['target'] + ' is ' + masks_str + '.')
            l_3 = list('Mu outlook on ' + stance_df['target'] + ' is ' + masks_str + '.')
            mixed_list = []
            for item1, item2, item3 in zip(l_1, l_2, l_3):
                selected_item = random.choice([item1, item2, item3])
                mixed_list.append(selected_item)
            return list(stance_df['text']), mixed_list
        else:
            raise ValueError(f'unknown prompt_type: {prompt_type}')
    if lang == 'pl':
        if prompt_type == 1:
            return list(stance_df['text']), \
                list('Moja postawa w kierunku ' + stance_df['target'] + ' jest: ' + masks_str + '.')
        elif prompt_type == 2:
            return list(stance_df['text']), \
                list('Moja postawa w kierunku ' + stance_df['target'] + ' jest ' + masks_str + '.')
        elif prompt_type == 3:
            return list(stance_df['text']), \
                list('Więc moja postawa w kierunku ' + stance_df['target'] + ' jest: ' + masks_str + '.')
        elif prompt_type == 4:
            return list(stance_df['text']), \
                list('Więc moja postawa w kierunku ' + stance_df['target'] + ' jest ' + masks_str + '.')
        else:
            raise ValueError(f'unknown prompt_type: {prompt_type}')


def tokenize_input(input_df, tokenizer, max_length=None, definitions=False,
                   multi_cls_2=False, two_new_tokens=False, masked_lm=False,
                   masked_lm_tokens=1, masked_lm_prompt=4, masked_lm_single_seq=False,
                   lang='en', masked_lm_new_token=False):
    if multi_cls_2 and two_new_tokens:
        raise ValueError('multi_cls_2 and multi_cls_4 cannot be both True')
    masked_lm_new_token_str = None
    if masked_lm_new_token:
        masked_lm_new_token_str = '[MASKNEW]'
        tokenizer.add_tokens([masked_lm_new_token_str, ])
    if definitions:
        assert not multi_cls_2 and not two_new_tokens
        if masked_lm:
            prompt_text, prompt_target = format_masked_lm_prompt(input_df, masked_lm_tokens, tokenizer,
                                                                 masked_lm_prompt,
                                                                 new_token=masked_lm_new_token_str)
            stance_col = pd.Series(prompt_text) + ' ' + pd.Series(prompt_target)
        else:
            stance_col = 'Text: ' + input_df['text'] + ' Target: ' + input_df['target']
        return tokenizer(list(stance_col), list(input_df['definition']),
                         truncation='only_second', padding=True, max_length=max_length, return_tensors='pt')
    elif masked_lm:
        prompt_text, prompt_target = format_masked_lm_prompt(input_df, masked_lm_tokens, tokenizer,
                                                             masked_lm_prompt, lang=lang,
                                                             new_token=masked_lm_new_token_str)
        if masked_lm_single_seq:
            stance_col = pd.Series(prompt_text) + ' ' + pd.Series(prompt_target)
            tokenizer(list(stance_col), list(input_df['definition']),
                      truncation='only_second', padding=True, max_length=max_length, return_tensors='pt')
        else:
            return tokenizer(prompt_text, prompt_target, truncation='only_first', padding=True, max_length=max_length,
                             return_tensors='pt')
    else:
        if two_new_tokens:
            tokenizer.add_tokens(['[CLS1]', '[CLS2]'])
            text_col = '[CLS1] ' + input_df['text']
            target_col = tokenizer.cls_token + ' [CLS2] ' + input_df['target']
        elif multi_cls_2:
            text_col = input_df['text']
            target_col = tokenizer.cls_token + ' ' + input_df['target']
        else:
            text_col = input_df['text']
            target_col = input_df['target']
        return tokenizer(list(text_col), list(target_col),
                         truncation='only_first', padding=True, max_length=max_length, return_tensors='pt')


def format_seq2seq_input(target, text, method='stancetarget_document'):
    if method == 'stancetarget_document':
        return f"stance target: {target} document: {text}"
    else:
        raise ValueError('wrong method')


def tokenize_seq2seq_input(input_df, tokenizer, max_length=280, label_max_len=2,
                           formatting_method='stancetarget_document', definitions=False):
    if definitions:
        raise NotImplementedError('not implemented yet')

    texts = [format_seq2seq_input(target, text, method=formatting_method) for target, text in
             zip(input_df['target'], input_df['text'])]

    tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt',
                          max_length=max_length)
    tokenized['labels'] = tokenizer(list(input_df['stance']), return_tensors='pt', max_length=label_max_len)[
        'input_ids']

    return tokenized


def get_datasets_dfs(vast=False, semeval=False, covid=False, vast_pl=False, semeval_zeroshot=None, clarin_pl=False,
                     clarin_zeroshot=None, semeval_80_20=None, vast_alt=None,
                     # generated datasets
                     vast_gen_1=False, vast_gen_2=False, vast_gen_3=False, vast_gen_4=False, vast_gen_s_1=False,
                     gen_tt_pl_1=False, gen_tt_pl_1_short=False, vast_gpt_1=False, seval_gpt_1=False,
                     # corrected datasets
                     vast_t3=False,
                     # vast options
                     vast_topic_only=False, vast_preprocess_text=False, vast_filter_ambiguous=False,
                     vast_filter_type_3=False, vast_preprocess_topic=False, vast_only_true_neutral=False,
                     gpt_synth_neutral_only=False, gpt_multi_only=True,
                     # semeval options
                     semeval_zeroshot_eval_on_test=False, semeval_exclude_dt=True, semeval_single_target=None,
                     semeval_adapt_target_names=False, semeval_adapt_target_names_type='lowercase',
                     semeval_remove_hashtags=False, semeval_zeroshot_test_only=False,
                     semeval_zeroshot_augment_targets=False, semeval_zs_rename_cl=False,
                     # generated datasets options
                     gen_neutral_synth_only=False, gen_tt_pl_favor_only=False,
                     # common options
                     limit_samples=None):
    """
    vast, semeval, covid, vast_pl etc. - whether to use this dataset. If multiple selected, sets will be merged
        (stacking train, dev and test separately). Also array of values 'train', 'dev' or 'test' may be used, to use
        only specified subsets e.g. vast = ['train', 'dev'] will only use train and dev sets from vast
    """

    train_dfs = []
    dev_dfs = []
    test_dfs = []
    if vast:
        train_df, dev_df, test_df = load_vast_dataset(contains_topic_only=vast_topic_only,
                                                      use_preprocessed_topic=vast_preprocess_topic,
                                                      use_preprocessed_post=vast_preprocess_text,
                                                      filter_ambiguous=vast_filter_ambiguous,
                                                      filter_type_3=vast_filter_type_3,
                                                      only_true_neutral=vast_only_true_neutral)
        if vast is True or 'train' in vast:
            train_dfs.append(train_df)
        if vast is True or 'dev' in vast:
            dev_dfs.append(dev_df)
        if vast is True or 'test' in vast:
            test_dfs.append(test_df)

    if vast_alt:
        train_df, dev_df, test_df = load_vast_dataset(train_path='data/vast_alt_train.csv',
                                                      dev_path='data/vast_alt_dev.csv',
                                                      test_path='data/vast_alt_test.csv',
                                                      contains_topic_only=vast_topic_only,
                                                      use_preprocessed_topic=vast_preprocess_topic,
                                                      use_preprocessed_post=vast_preprocess_text,
                                                      filter_ambiguous=vast_filter_ambiguous,
                                                      filter_type_3=vast_filter_type_3, )
        if vast_alt is True or 'train' in vast_alt:
            train_dfs.append(train_df)
        if vast_alt is True or 'dev' in vast_alt:
            dev_dfs.append(dev_df)
        if vast_alt is True or 'test' in vast_alt:
            test_dfs.append(test_df)

    if vast_gpt_1:
        train_df, dev_df = load_gpt_dataset('gpt_vast_1',
                                            synth_neutral_only=gpt_synth_neutral_only,
                                            multi_only=gpt_multi_only)
        if vast_gpt_1 is True or 'train' in vast_gpt_1:
            train_dfs.append(train_df)
        if vast_gpt_1 is True or 'dev' in vast_gpt_1:
            dev_dfs.append(dev_df)

    if seval_gpt_1:
        train_df, dev_df = load_gpt_dataset('gpt_seval_1',
                                            synth_neutral_only=gpt_synth_neutral_only,
                                            multi_only=gpt_multi_only)
        if seval_gpt_1 is True or 'train' in seval_gpt_1:
            train_dfs.append(train_df)
        if seval_gpt_1 is True or 'dev' in seval_gpt_1:
            dev_dfs.append(dev_df)

    if semeval:
        train_df, dev_df, test_df = load_semeval_dataset(exclude_dt=semeval_exclude_dt,
                                                         single_target=semeval_single_target,
                                                         adapt_target_names=semeval_adapt_target_names,
                                                         adapt_target_names_type=semeval_adapt_target_names_type,
                                                         remove_hashtags=semeval_remove_hashtags)
        if semeval is True or 'train' in semeval:
            train_dfs.append(train_df)
        if semeval is True or 'dev' in semeval:
            dev_dfs.append(dev_df)
        if semeval is True or 'test' in semeval:
            test_dfs.append(test_df)

    if semeval_80_20:
        train_df, dev_df, _ = load_semeval_dataset(exclude_dt=False, train_val_80_20=True,
                                                   random_val_dataset=False,
                                                   adapt_target_names=semeval_adapt_target_names,
                                                   adapt_target_names_type=semeval_adapt_target_names_type,
                                                   remove_hashtags=semeval_remove_hashtags)
        if semeval_80_20 is True or 'train' in semeval_80_20:
            train_dfs.append(train_df)
        if semeval_80_20 is True or 'dev' in semeval_80_20:
            dev_dfs.append(dev_df)

    if covid:
        train_df, dev_df, test_df = load_covid_dataset()
        if covid == True or 'train' in covid:
            train_dfs.append(train_df)
        if covid == True or 'dev' in covid:
            dev_dfs.append(dev_df)
        if covid == True or 'test' in covid:
            test_dfs.append(test_df)

    if vast_pl:
        train_df, dev_df, test_df = load_vast_dataset(
            train_path='data/vast_train_auto_pl.csv',
            dev_path='data/vast_dev_auto_pl.csv',
            test_path='data/vast_test_auto_pl.csv',
            polish=True, contains_topic_only=vast_topic_only,
            use_preprocessed_topic=vast_preprocess_text,
            use_preprocessed_post=vast_preprocess_text,
            filter_ambiguous=vast_filter_ambiguous,
            filter_type_3=vast_filter_type_3, )
        if vast_pl == True or 'train' in vast_pl:
            train_dfs.append(train_df)
        if vast_pl == True or 'dev' in vast_pl:
            dev_dfs.append(dev_df)
        if vast_pl == True or 'test' in vast_pl:
            test_dfs.append(test_df)

    if vast_t3:
        train_df = load_vast_t3_dataset()
        if vast_t3 is True or 'train' in vast_t3:
            train_dfs.append(train_df)

    if semeval_zeroshot:
        train_df, dev_df, test_df = load_semeval_zeroshot_dataset(semeval_zeroshot,
                                                                  eval_on_test=semeval_zeroshot_eval_on_test,
                                                                  adapt_target_names=semeval_adapt_target_names,
                                                                  remove_hashtags=semeval_remove_hashtags,
                                                                  augment_targets=semeval_zeroshot_augment_targets,
                                                                  rename_cl=semeval_zs_rename_cl)
        if semeval_zeroshot is not None:
            if not semeval_zeroshot_test_only:
                train_dfs.append(train_df)
                dev_dfs.append(dev_df)
            test_dfs.append(test_df)

    if vast_gen_1:
        train_df = load_gen_vast_1_dataset(gen_neutral_synth_only=gen_neutral_synth_only)
        if vast_gen_1 == True or 'train' in vast_gen_1:
            train_dfs.append(train_df)

    if vast_gen_2:
        train_df = load_gen_vast_2_dataset(gen_neutral_synth_only=gen_neutral_synth_only)
        if vast_gen_2 == True or 'train' in vast_gen_2:
            train_dfs.append(train_df)

    if vast_gen_3:
        train_df = load_gen_vast_3_dataset(gen_neutral_synth_only=gen_neutral_synth_only)
        if vast_gen_3 == True or 'train' in vast_gen_3:
            train_dfs.append(train_df)

    if vast_gen_4:
        train_df = load_gen_vast_4_dataset(gen_neutral_synth_only=gen_neutral_synth_only)
        if vast_gen_4 == True or 'train' in vast_gen_4:
            train_dfs.append(train_df)

    if vast_gen_s_1:
        train_df = load_gen_vast_s_1_dataset(gen_neutral_synth_only=gen_neutral_synth_only)
        if vast_gen_s_1 == True or 'train' in vast_gen_s_1:
            train_dfs.append(train_df)

    if clarin_pl:
        train_df, dev_df = load_clarin_dataset()
        if clarin_pl == True or 'train' in clarin_pl:
            train_dfs.append(train_df)
        if clarin_pl == True or 'dev' in clarin_pl:
            dev_dfs.append(dev_df)
        if clarin_pl == True or 'test' in clarin_pl:
            test_dfs.append(dev_df)

    if clarin_zeroshot:
        train_df, dev_df, test_df = load_clarin_zs_dataset(clarin_zeroshot)
        if clarin_zeroshot is not None:
            train_dfs.append(train_df)
            dev_dfs.append(dev_df)
            test_dfs.append(test_df)

    if gen_tt_pl_1:
        train_df, dev_df = load_gen_tt_pl_1_dataset(fovor_only=gen_tt_pl_favor_only, short_target_only=False)
        if gen_tt_pl_1 == True or 'train' in gen_tt_pl_1:
            train_dfs.append(train_df)
        if gen_tt_pl_1 == True or 'dev' in gen_tt_pl_1:
            dev_dfs.append(dev_df)

    if gen_tt_pl_1_short:
        train_df, dev_df = load_gen_tt_pl_1_dataset(fovor_only=gen_tt_pl_favor_only, short_target_only=True)
        if gen_tt_pl_1_short == True or 'train' in gen_tt_pl_1_short:
            train_dfs.append(train_df)
        if gen_tt_pl_1_short == True or 'dev' in gen_tt_pl_1_short:
            dev_dfs.append(dev_df)

    train_df = pd.concat(train_dfs, ignore_index=True) if len(train_dfs) > 0 else None
    dev_df = pd.concat(dev_dfs, ignore_index=True) if len(dev_dfs) > 0 else None
    test_df = pd.concat(test_dfs, ignore_index=True) if len(test_dfs) > 0 else None

    if seval_gpt_1:
        print('filtering out test samples from train and dev')
        len_before = len(train_df) + len(dev_df)
        train_df = train_df[~train_df['text'].isin(test_df['text'].unique())]
        dev_df = dev_df[~dev_df['text'].isin(test_df['text'].unique())]
        print(f'filtered out {len_before - len(train_df) - len(dev_df)} samples')

    if limit_samples:
        train_df = train_df.sample(limit_samples) if train_df is not None else None
        dev_df = dev_df.sample(limit_samples) if dev_df is not None else None
        test_df = test_df.sample(limit_samples) if test_df is not None else None

    return train_df, dev_df, test_df


def _create_concat_augemented_df(df: pd.DataFrame):
    max_len = int(np.percentile(df['text'].str.len(), 50))
    source_df = df[df['text'].str.len() <= max_len].copy()
    for i, row in source_df.iterrows():
        concat_text = ''
        for j, row_j in source_df.sample(frac=1.0).iterrows():
            if i == j:
                continue
            target_simil = fuzz.ratio(row['target'], row_j['target'])
            if target_simil <= 50:
                concat_text = row_j['text']
                break
        if concat_text == '':
            logger.warning(f'training concatenation augmentation: no dissimilar target found, row {i}')
        if random.random() < 0.5:
            concat_text = row['text'] + ' ' + concat_text
        else:
            concat_text = concat_text + ' ' + row['text']
        source_df.at[i, 'text'] = concat_text
    return source_df


def get_datasets(model_name_or_path: str, target_definitions_file=None, max_len=280,
                 multi_cls_2=False, two_new_tokens=False,
                 masked_lm=False, masked_lm_tokens=1, masked_lm_prompt=4, masked_lm_lang='en',
                 masked_lm_new_token=False,
                 zeroshot_test=False,
                 concat_augment_train=False,
                 limit_train_samples=None,
                 remove_punctuation_jcl=False,
                 target_word_limit=None,
                 lowercase_targets=False,
                 seed=1,
                 **kwargs):
    """

    :param model_name_or_path: transformers model name or path
    :param target_definitions_file: path to a file with target definitions
    :param max_len: max length of sequence for the tokenizer
    :param multi_cls_2: use additional token before stance target
    :param two_new_tokens: add 2 new tokens to the tokenizer
    and place first before text and second before stance target
    :param masked_lm: use masked language modeling (prompt) mode
    :param masked_lm_tokens: number of mask tokens added to the prompt
    :param masked_lm_prompt: masked lm prompt number
    :param masked_lm_lang: masked lm prompt language
    :param masked_lm_new_token: in masked lm prompt use new token instead of [MASK]
    :param zeroshot_test: use zero-shot test dataset (if available for specified datasets)
    :param concat_augment_train: data augmentation by concatenating samples with dissimilar targets
    :param limit_train_samples: limit number of train samples
    :param seed: seed for limiting train samples

    Returns:
        dict: A dictionary with the following keys:
            datasets (tuple): (train, dev, test),
            zeroshot_test_dataset: zero-shot test dataset (if available and requested),
            special_tokens: tokens added to tokenizer,
            mask_token_id: id of the mask token for used tokenizer,
            verbalizer_token_ids: ids of verbalizer tokens for prompts (see `DEFAULT_VERBALIZERS`),
            dfs: dataframes with train, dev, test data,
            tokenizer used tokenizer

    """

    train_df, dev_df, test_df = get_datasets_dfs(**kwargs)

    if train_df is None:
        raise ValueError('missing train data')
    if dev_df is None:
        raise ValueError('missing dev data')
    if test_df is None:
        logger.warning('missing test data')

    if concat_augment_train:
        augmented_train = _create_concat_augemented_df(train_df)
        logger.info(f'augmented train size: {len(augmented_train)}')
        train_df = pd.concat([train_df, augmented_train])

    if limit_train_samples:
        train_df = train_df.sample(limit_train_samples, random_state=seed)

    logger.info(f'train size: {len(train_df) if train_df is not None else 0}')
    logger.info(f'unique targets in train: {len(train_df["target"].unique()) if train_df is not None else 0}')
    logger.info(f'dev size: {len(dev_df) if dev_df is not None else 0}')
    logger.info(f'test size: {len(test_df) if test_df is not None else 0}')
    logger.info(f'train class proportions:\n{train_df["class"].value_counts(normalize=True).sort_index()}')

    if remove_punctuation_jcl:
        for df in [train_df, dev_df, test_df]:
            if df is not None:
                df['text'] = df['text'].apply(_remove_punctuation_jcl)

    if target_word_limit:
        for df in [train_df, dev_df, test_df]:
            if df is not None:
                df['target'] = df['target'].apply(lambda x: ' '.join(x.split()[:target_word_limit]))

    if lowercase_targets:
        for df in [train_df, dev_df, test_df]:
            if df is not None:
                df['target'] = df['target'].apply(lambda x: x.lower())

    if target_definitions_file:
        train_df = add_target_definitions_to_df(train_df, target_definitions_file) if train_df is not None else None
        dev_df = add_target_definitions_to_df(dev_df, target_definitions_file) if dev_df is not None else None
        test_df = add_target_definitions_to_df(test_df, target_definitions_file) if test_df is not None else None

    if 'seen' not in test_df.columns:
        zeroshot_test_df = None
    else:
        zeroshot_test_df = test_df[test_df['seen'].fillna(1).astype(int) == 0] if zeroshot_test else None

    if 'type_idx' not in test_df.columns:
        true_neutral_test_df = None
    else:
        true_neutral_test_df = test_df[test_df['type_idx'] != 4]


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    special_tokens = None

    if model_name_or_path in _SEQ2SEQ_TOKENIZERS:
        train_encodings = tokenize_seq2seq_input(train_df, tokenizer,
                                                 max_length=max_len) if train_df is not None else None
        dev_encodings = tokenize_seq2seq_input(dev_df, tokenizer, max_length=max_len) if dev_df is not None else None
        test_encodings = tokenize_seq2seq_input(test_df, tokenizer, max_length=max_len) if test_df is not None else None
        zs_test_encodings = tokenize_seq2seq_input(zeroshot_test_df, tokenizer,
                                                   max_length=max_len) if zeroshot_test_df is not None else None
        true_neutral_test_encodings = tokenize_seq2seq_input(true_neutral_test_df, tokenizer,
                                                             max_length=max_len) if true_neutral_test_df is not None else None
        train_dataset = StanceDataset(train_encodings) if train_encodings is not None else None
        dev_dataset = StanceDataset(dev_encodings) if dev_encodings is not None else None
        test_dataset = StanceDataset(test_encodings) if test_encodings is not None else None
        zs_test_dataset = StanceDataset(zs_test_encodings) if zs_test_encodings is not None else None
        true_neutral_test_dataset = StanceDataset(true_neutral_test_encodings) if true_neutral_test_encodings is not None else None
    else:
        definitions = True if target_definitions_file else False
        init_tokenizer_len = len(tokenizer)

        all_datasets = []
        split_names = ['train', 'dev', 'test', 'zeroshot_test', 'true_neutral_test']
        for current_df, split_name in zip([train_df, dev_df, test_df, zeroshot_test_df, true_neutral_test_df], split_names):
            if current_df is None:
                all_datasets.append(None)
                continue
            current_encodings = tokenize_input(current_df, tokenizer, definitions=definitions,
                                               max_length=max_len, multi_cls_2=multi_cls_2,
                                               two_new_tokens=two_new_tokens, masked_lm=masked_lm,
                                               masked_lm_tokens=masked_lm_tokens,
                                               masked_lm_prompt=masked_lm_prompt,
                                               lang=masked_lm_lang,
                                               masked_lm_new_token=masked_lm_new_token)
            logger.info(
                f'max tokens in {split_name}: {np.max([np.array(e.attention_mask).sum() for e in current_encodings.encodings])}')
            current_dataset = StanceDataset(current_encodings, list(current_df['class']))
            all_datasets.append(current_dataset)

        train_dataset, dev_dataset, test_dataset, zs_test_dataset, true_neutral_test_dataset = all_datasets

        final_tokenizer_len = len(tokenizer)
        if final_tokenizer_len > init_tokenizer_len:
            special_tokens = list(range(init_tokenizer_len, final_tokenizer_len))

    verbalizer_token_ids = []
    for verbalizer in DEFAULT_VERBALIZERS:
        encoded = tokenizer.encode(verbalizer, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError('expected verbalizer to be a single token')
        verbalizer_token_ids.append(encoded[0])

    return {'datasets': (train_dataset, dev_dataset, test_dataset),
            'zeroshot_test_dataset': zs_test_dataset,
            'true_neutral_test_dataset': true_neutral_test_dataset,
            'special_tokens': special_tokens,
            'mask_token_id': tokenizer.mask_token_id,
            'verbalizer_token_ids': verbalizer_token_ids,
            'dfs': (train_df, dev_df, test_df),
            'tokenizer': tokenizer}
