import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from tqdm import tqdm




def get_macro_f1(pair_df, underlying_property_col='category', suffix=('_1', '_2'), pred_col_name='pred', label_col_name='label'):
    suff_1 = suffix[0]
    suff_2 = suffix[1]
    label_vals = pair_df[underlying_property_col + suff_1].unique()
    F1s = {}
    pair_df = pair_df[pair_df['id'+suff_1] < pair_df['id'+suff_2]]  # remove duplicates
    for val in label_vals:
        rel_df = pair_df[(pair_df[underlying_property_col + suff_1] == val) | (pair_df[underlying_property_col + suff_2] == val)]
        # rel_df = pair_df[(pair_df[underlying_property_col + suff_1] == val)] # testing without two directional matching
        if len(rel_df) == 0:
            print(f"ERR. No pairs with label {val} in the dataframe")
            continue
        F1s[val] = f1_score(rel_df[label_col_name], rel_df[pred_col_name])
    return np.mean(list(F1s.values())), F1s
        


