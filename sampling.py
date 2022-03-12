import os
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

input_dir = "/home/idomino/OU/t847/data/processed/"
output_dir = "/home/idomino/OU/t847/data/samples_new/"

seed_list = [1337, 1338, 1339]

for i in range(0,20):
    print(f'Reading dataframe #{i}...')
    df = pd.read_pickle(input_dir + f'data{i}.pkl', compression='zip')
    
    meta_df = df[['appeared','label','avclass']].copy()
    meta_df['index'] = meta_df.index
    meta_df['fragment'] = i
    
    mode = 'w' if i==0 else 'a'
    header = True if i==0 else False
    meta_df.to_csv(output_dir + 'metadata.csv', index=False, mode=mode, header=header)
    del df

metadata = pd.read_csv(output_dir + 'metadata.csv')
metadata.appeared = pd.to_datetime(metadata.appeared)

malware_mask = np.logical_and(metadata.avclass != '-',metadata.label == 1)
benign_mask = (metadata.label == 0)

first_malware_time = pd.Timestamp('2018-01-01 00:00:00')
split_time = pd.Timestamp('2018-07-31 00:00:00')

train_mask = np.logical_and(metadata.appeared >= first_malware_time, metadata.appeared < split_time)
test_mask = metadata.appeared > split_time

[np.logical_and(malware_mask,train_mask)].avclass.value_counts()[0:10]

metadata[np.logical_and(malware_mask,test_mask)].avclass.value_counts()[0:10]

top_n = 50
train_families = set(metadata[np.logical_and(malware_mask,train_mask)].avclass.value_counts()[0:top_n].index)
test_families = set(metadata[np.logical_and(malware_mask,test_mask)].avclass.value_counts()[0:top_n].index)
intersect_families = train_families.intersection(test_families)
print('Intersection families:', intersect_families)
print()
intersect_mask = metadata.avclass.apply(lambda x: x in intersect_families)

train_malware_samples = metadata[np.logical_and(malware_mask,np.logical_and(train_mask,intersect_mask))]
print('Train malware samples:', train_malware_samples.shape[0])

test_malware_samples = metadata[np.logical_and(malware_mask,np.logical_and(test_mask,intersect_mask))]
print('Test malware samples:', test_malware_samples.shape[0])

train_benign_samples = metadata[np.logical_and(benign_mask,train_mask)]
print('Train benign samples:', train_benign_samples.shape[0])

test_benign_samples = metadata[np.logical_and(benign_mask,test_mask)]
print('Test benign samples:', test_benign_samples.shape[0])

def prepare_samples(samples_for, n_malware, ratio_benign, seed):
    
    n_benign = n_malware * ratio_benign
    msg = f'[s{seed}] Preapring {samples_for} file of {n_malware} malware / {n_benign} benignware (1:{ratio_benign})...'
    print(msg)
    
    malware_pool = None
    benign_pool = None

    if samples_for == 'train':
        malware_pool = train_malware_samples
        benign_pool = train_benign_samples
    elif samples_for == 'test':
        malware_pool = test_malware_samples
        benign_pool = test_benign_samples
    else:
        raise Exception('Invalid "sample_for" value, should be "train" or "test"!')

    malware_picked = malware_pool.sample(n_malware, random_state=seed)
    benign_picked = benign_pool.sample(n_benign, random_state=seed)

    acc_df = None
    for i in range(0,20):
        print(f'Reading dataframe #{i}...')
        df = pd.read_pickle(input_dir + f'data{i}.pkl', compression='zip')

        malware_idx = list(malware_picked[malware_picked.fragment == i]['index'])
        benign_idx = list(benign_picked[benign_picked.fragment == i]['index'])
        idx = malware_idx + benign_idx

        if acc_df is not None:
            acc_df = pd.concat([acc_df,df.loc[idx].copy()])
        else:
            acc_df = df.loc[idx].copy()

        del df

    core_columns = ['appeared', 'label', 'avclass']
    feature_columns = [x for x in acc_df.columns if x not in core_columns]
    format_agnostic_columns = [x for x in feature_columns 
                                   if x.startswith('histogram') 
                                   or x.startswith('byteentropy') 
                                   or x.startswith('strings')]
    parsed_columns = [x for x in feature_columns if x not in format_agnostic_columns]

    format_agnostic_columns = format_agnostic_columns + core_columns
    parsed_columns = parsed_columns + core_columns

    format_agnostic_df = acc_df[format_agnostic_columns]
    parsed_df = acc_df[parsed_columns]

    filename = f'{samples_for}_{n_malware}_malware_x{ratio_benign}_benign_format_agnostic_s{seed}.pkl'
    format_agnostic_df.to_pickle(output_dir + filename, compression='zip')
    print('Saved:', filename)

    filename = f'{samples_for}_{n_malware}_malware_x{ratio_benign}_benign_parsed_s{seed}.pkl'
    parsed_df.to_pickle(output_dir + filename, compression='zip')
    print('Saved:', filename)
    
    filename = f'{samples_for}_{n_malware}_malware_x{ratio_benign}_benign_combined_s{seed}.pkl'
    acc_df.to_pickle(output_dir + filename, compression='zip')
    print('Saved:', filename)

required_samples = [
    ('train', 100, 1),
    ('train', 200, 1),
    ('train', 400, 1),
    ('train', 800, 1),
    ('train', 1600, 1),
    ('train', 3200, 1),
    ('train', 6400, 1),
    ('train', 12800, 1),
    ('train', 25600, 1),
    ('train', 51200, 1),
    ('train', 102400, 1),
    
    ('test', 100, 1),
    ('test', 200, 1),
    ('test', 400, 1),
    ('test', 800, 1),
    ('test', 1600, 1),
    ('test', 3200, 1),
    ('test', 6400, 1),
    ('test', 12800, 1),
    ('test', 25600, 1),
    ('test', 51200, 1),
    ('test', 102400, 1),
    
    ('test', 1250, 1),
    ('test', 1250, 2),
    ('test', 1250, 4),
    ('test', 1250, 8),
    ('test', 1250, 16),
    ('test', 1250, 32),
    ('test', 1250, 64),
    ('test', 1250, 128)
]

for sample_params in required_samples:
    for s in seed_list:
        prepare_samples(*sample_params, s)
