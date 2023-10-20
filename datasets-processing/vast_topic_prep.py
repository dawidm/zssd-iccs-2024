from stancedetect.stancedatasets import *

vast_train_raw, vast_dev_raw, vast_test_raw = load_vast_dataset(adapt=False, add_stance_class=False)

def prep_4_topics_in_vast(vast_raw_df):
    vast_raw_df = vast_raw_df.copy()
    vast_raw_df['new_topic_corr'] = vast_raw_df['new_topic'].copy()
    vast_raw_df_type_4 = vast_raw_df[vast_raw_df['type_idx'] == 4]
    vast_raw_df_type_not4 = vast_raw_df[vast_raw_df['type_idx'] != 4]

    for i in vast_raw_df_type_4.index:
        topic_prep = vast_raw_df_type_4.loc[i,'topic_str']
        topic_new = vast_raw_df_type_not4[vast_raw_df_type_not4['topic_str'] == topic_prep].iloc[0]['new_topic']
        vast_raw_df.loc[i,'new_topic_corr'] = topic_new

    return vast_raw_df
    
prep_train = prep_4_topics_in_vast(vast_train_raw)
prep_dev = prep_4_topics_in_vast(vast_dev_raw)
prep_test = prep_4_topics_in_vast(vast_test_raw)

prep_train.to_csv('data/vast_train_corr.csv', index=None)
prep_dev.to_csv('data/vast_dev_corr.csv', index=None)
prep_test.to_csv('data/vast_test_corr.csv', index=None)
