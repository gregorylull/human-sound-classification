# (glull) file is used to massage the data from the Chime project.

# this constant is primarily used to debug,
# to run this file and extract all the audio samples, set this limit to None.
MAX_FILES_TO_EXTRACT = 5 # None

# At this point i do not know how to get the root folder programmatically,
# if this file/folder is moved around the rel path needs to be updated
ROOT_FOLDER = '../../'

import glob
import pandas as pd
import numpy as np
import re
import pickle
import time

# my files
from src.utilities import feature_extraction as glfex

# constants
data_folder = f'{ROOT_FOLDER}data/chime_home/'
data_chunks = f'{data_folder}chunks/'
dev_chunks_refined_filename = 'development_chunks_refined.csv'

eval_dev_chunks = glob.glob(f'{data_folder}*.csv')
raw_chunks = glob.glob(f'{data_chunks}*.csv')

def get_all_csv_files(filenames):
    """
    input:
        filenames - list of filepaths from glob
    
    output:
        a dictionary of dataframes, e.g.:
        {
            'development_chunks_refined.csv': pd.DataFrame
        }

    """
    files = {}
    for filename in filenames:
        filename_without_path = filename.split('/')[-1]

        files[filename_without_path] = pd.read_csv(
            filename,
            header=None,
            names=['index', 'filename']
        )
    
    return files

def get_chunks_df(dataframe, filenames):
    """
    For this dataset there are multiple .csv files that relate to which chunks contain labeled data
    """
    results = []
    for filename in filenames:
        chunk_df = pd.read_csv(f'{data_chunks}{filename}.csv', header=None)
        chunk_t = chunk_df.set_index(0).T
        results.append(chunk_t)
        
    concat_df = pd.concat(results)
    return concat_df

def add_human_features(dataframe):
    """
    This dataset had three scientists label what sounds/noise/voices an audio sample contains, the 'majorityvote' column
    is a consensus of all three humans being sure what an audio sample contains
    """
    human_voices_re = re.compile(r'[cmf]') # child male female
    dataframe['human_voice'] = dataframe['majorityvote'].apply(lambda x: 1 if human_voices_re.search(x) else 0)

    female_voice = re.compile(r'[cmf]') # child male female
    dataframe['female_voice'] = dataframe['majorityvote'].apply(lambda x: 1 if female_voice.search(x) else 0)

    male_voice_re = re.compile(r'[cmf]') # child male female
    dataframe['male_voice'] = dataframe['majorityvote'].apply(lambda x: 1 if male_voice_re.search(x) else 0)

    child_voice = re.compile(r'[cmf]') # child male female
    dataframe['child_voice'] = dataframe['majorityvote'].apply(lambda x: 1 if child_voice.search(x) else 0)

    return dataframe

def extract_features_df(df, max=None, column_name = 'chunkname', codec='.48kHz.wav'):
    """
    Read an audio file and extract the mfcc features using librosa
    """
    results = {}
    total = df.shape[0]
    count = 0
    for index, row in df.iterrows():
        if max and count == max:
            break

        print('what is left', (count/total) * 100, '%')

        # print(time.ctime(), end="\r", flush=True)

        chunkname = row[column_name]
        chunk_filename = f'{data_chunks}{chunkname}{codec}'
        features = glfex.extract_features(chunk_filename)
        results[chunkname] = features
        
        count += 1
        
    return results

def merge_features(df, features):
    """
    The dataframe containing information on which chunkfiles contain human sounds is different than
    then the extracted features which is a dictionary. This func combines the two into a dataframe.
    """
    results = []
    
    for index, row in df.iterrows():
        chunkname = row['chunkname']
        if chunkname in features:
            feature_mfcc = features[chunkname]['mfcc']
            feature_mfcc_delta = features[chunkname]['mfcc_delta']
            feature_beat_mfcc_delta = features[chunkname]['beat_mfcc_delta']

            # feature_mfcc_flattened = np.array(feature_mfcc).flatten().tolist()
            # feature_mfcc_delta_flattened = np.array(feature_mfcc_delta).flatten().tolist()
            # feature_beat_mfcc_delta_flattened = np.array(feature_beat_mfcc_delta).flatten().tolist()

            feature_mfcc_flattened = glfex.flatten(feature_mfcc)
            feature_mfcc_delta_flattened = glfex.flatten(feature_mfcc_delta)
            feature_beat_mfcc_delta_flattened = glfex.flatten(feature_beat_mfcc_delta)

            flattened_mfcc_features = feature_mfcc_flattened + feature_mfcc_delta_flattened + feature_beat_mfcc_delta_flattened

            is_human = row['human_voice']
            is_female = row['female_voice']
            is_male = row['male_voice']
            is_child = row['child_voice']

            flattened_mfcc_features.insert(0, chunkname)
            flattened_mfcc_features.insert(0, is_human)
            flattened_mfcc_features.insert(0, is_female)
            flattened_mfcc_features.insert(0, is_male)
            flattened_mfcc_features.insert(0, is_child)

            results.append(flattened_mfcc_features)
    
    df_raw = pd.DataFrame(results)

    df_raw.rename(columns={
        0: 'has_child',
        1: 'has_male',
        2: 'has_female',
        3: 'has_human',
        4: 'chunkname'
    }, inplace=True)

    return df_raw


def main():
    dataframe_files = get_all_csv_files(eval_dev_chunks)

    # this list consists of a list of filename for different sound chunks 
    chunks_refined_df = dataframe_files[dev_chunks_refined_filename]
    chunks_refined_df = chunks_refined_df.drop(columns=['index'])

    # this list consists chunknames and some features such as sound environment classification
    # e.g.: female, child, male, noise
    refined_df = get_chunks_df(chunks_refined_df, chunks_refined_df['filename'])

    # has information on whether an audio sample contains human sounds
    human_features_df = add_human_features(refined_df)

    # extract features from audio samples, this contains ALL features, e.g.:
    # mfcc, mfcc_delta, chromagram, etc.
    mfcc_features = extract_features_df(human_features_df, MAX_FILES_TO_EXTRACT)
    features_pkl_filename = 'chime_mfcc_features_only.pkl'
    with open(features_pkl_filename, 'wb') as writefile:
        pickle.dump(mfcc_features, writefile)
    print(f'\nsaved all mfcc features to {features_pkl_filename}\n')

    # combines the human features db with ONLY the mfcc
    full_df = merge_features(human_features_df, mfcc_features)
    print('Extracted all features, shape:', full_df.shape)

    csv_filename = 'chime_mfcc.csv'

    full_df.to_csv(csv_filename)

    print(f'\nsaved data with target and features: mfcc, mfcc_delta, and beat_mfcc to: {csv_filename}\n')


if __name__ == '__main__':
    main()




