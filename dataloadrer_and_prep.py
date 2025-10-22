import torch
import pandas as pd
import numpy as np
import torch.utils
from loading_paper_utils import load_data_from_pkl
from torch.utils.data import DataLoader
from itertools import combinations
import random
from tqdm import tqdm
import random
from sentence_transformers import InputExample
from torch.utils.data import Dataset
import pickle as pkl
import os




def df_to_text_col_pairs(df, text_col_name, save_dir, format = 'single_csv'):
    print(f'Creating pairs from {text_col_name} column in dataframe of shape {df.shape}')
    print(df[text_col_name].head())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    categories = df['category'].values
    match_matrix_np = categories[:, None] == categories
    match_matrix_np = match_matrix_np.astype(int)
    index_pairs_of_0 = np.argwhere(np.triu(match_matrix_np, 1) == 0)
    # zero the diagonal
    index_pairs_of_1 = np.argwhere(np.triu(match_matrix_np,1) == 1)
    pos_pairs = [(df.iloc[i][text_col_name], df.iloc[j][text_col_name], 1, df.iloc[i]['id'], df.iloc[j]['id']) for i, j in index_pairs_of_1]
    # sample 3 negative pairs for each positive pair
    index_pairs_of_0 = np.random.permutation(index_pairs_of_0)[:len(pos_pairs) * 3]
    neg_pairs = [(df.iloc[i][text_col_name], df.iloc[j][text_col_name], 0, df.iloc[i]['id'], df.iloc[j]['id']) for i, j in index_pairs_of_0]
    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    print(f'Created {len(pairs)} pairs from {text_col_name} column')
    print('started saving pairs')
    if format == 'pkl':
        for idx, pair_label in tqdm(enumerate(pairs), total=len(pairs)):
            with open(f'{save_dir}/pair{idx}.pkl', 'wb') as f:
                pkl.dump(pair_label, f)
    elif format == 'csv':
        for idx, pair_label in tqdm(enumerate(pairs), total=len(pairs)):
            with open(f'{save_dir}/pair{idx}.csv', 'w') as f:
                f.write(f'{pair_label[0]},{pair_label[1]},{pair_label[2]}')
    elif format == 'single_csv':
        print('creating dataframe from pairs')
        print(f'Number of pairs: {len(pairs)}')
        # print(pairs[:5])  # print first 5 pairs for debugging
        data_df = pd.DataFrame(pairs, columns=['text1', 'text2', 'label', 'id1', 'id2'])
        print(data_df.columns)
        print(f'saving dataframe to {save_dir}/pairs.csv')
        data_df.to_csv(f'{save_dir}/pairs.csv', index=False)
    else:
        raise ValueError(f'format {format} not currently supported. Please use pkl or csv.')


def pkl_to_text_col_pairs(path_to_data_pkl, text_col_name, save_dir, format = 'single_csv'):
    # delete the save_dir if it exists
    os.makedirs(save_dir, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    # os.makedirs(save_dir, exist_ok=True)
    df = load_data_from_pkl(path_to_data_pkl).reset_index(drop=True)
    categories = df['category'].values
    match_matrix_np = categories[:, None] == categories
    match_matrix_np = match_matrix_np.astype(int)
    index_pairs_of_0 = np.argwhere(np.triu(match_matrix_np, 1) == 0)
    # zero the diagonal
    index_pairs_of_1 = np.argwhere(np.triu(match_matrix_np,1) == 1)
    pos_pairs = [(df.iloc[i][text_col_name], df.iloc[j][text_col_name], 1, df.iloc[i]['id'], df.iloc[j]['id']) for i, j in index_pairs_of_1]
    # sample 3 negative pairs for each positive pair
    index_pairs_of_0 = np.random.permutation(index_pairs_of_0)[:len(pos_pairs) * 3]
    neg_pairs = [(df.iloc[i][text_col_name], df.iloc[j][text_col_name], 0, df.iloc[i]['id'], df.iloc[j]['id']) for i, j in index_pairs_of_0]
    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    if format == 'pkl':
        for idx, pair_label in tqdm(enumerate(pairs), total=len(pairs)):
            with open(f'{save_dir}/pair{idx}.pkl', 'wb') as f:
                pkl.dump(pair_label, f)
    elif format == 'csv':
        for idx, pair_label in tqdm(enumerate(pairs), total=len(pairs)):
            with open(f'{save_dir}/pair{idx}.csv', 'w') as f:
                f.write(f'{pair_label[0]},{pair_label[1]},{pair_label[2]}')
    elif format == 'single_csv':
        data_df = pd.DataFrame(pairs, columns=['text1', 'text2', 'label', 'id1', 'id2'])
        data_df.to_csv(f'{save_dir}/pairs.csv', index=False)
    else:
        raise ValueError(f'format {format} not currently supported. Please use pkl or csv.')



def train_test_split(all_data_df, path_to_train_save_pkl, path_to_test_save_pkl):
    
    # samples 80% from each category for train and 20% for test
    train_df = all_data_df.groupby('category').apply(lambda x: x.sample(frac=0.8, random_state=42)).reset_index(drop=True)
    train_ids = train_df['id'].unique()
    test_df = all_data_df[~all_data_df['id'].isin(train_ids)].reset_index(drop=True)
    print(train_df.shape, test_df.shape)
    print(train_df['category'].value_counts())
    print(test_df['category'].value_counts())
    train_df.to_pickle(path_to_train_save_pkl)
    test_df.to_pickle(path_to_test_save_pkl)



def train_df_to_train_val_split(train_df, train_no_val_save_path, val_df_save_path):
    val_df = train_df.groupby('category').apply(lambda x: x.sample(frac=0.1, random_state=42)).reset_index(drop=True)
    val_df_ids = val_df['id'].unique()
    train_df_no_val = train_df[~train_df['id'].isin(val_df_ids)].reset_index(drop=True)
    train_df_no_val.to_pickle(train_no_val_save_path)
    val_df.to_pickle(val_df_save_path)

def main():
    pass

if __name__ == '__main__':
    main()

 
