import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from statistics import mode
from scipy.stats import entropy
from scipy.special import softmax
from matching_metrics import get_macro_f1
import random



"""
Process:
1. Learn abstract embeddings by finetuning a model on the train set.
REPEAT:
2. Use learned embeddings and transitivity of train labels to infer centroids for each equivalence class.
3. Use algorithm to select new representative paragraphs in the train set.
4. Evaluate the new reps with the best
5. Learn new embeddings for the selected paragraphs.
6. Go to step 2.
________________________________

"""






def train_df_to_centroids_v2(df, embedding_col_name, id_col_name='id', col_for_label='category'):
    # centroid extraction process, using labels for simplicity, this method can be adapted to use binary predictions and transitivity. 
    df = df.copy()
    grouped = df.groupby(col_for_label)
    centroids = []
    conv_dict = {}
    for i, (label, group) in enumerate(grouped):
        if len(group) > 0:
            centroid = group[embedding_col_name].mean(axis=0)
            centroids.append(centroid)
            conv_dict[label] = i
    centroids = np.array(centroids)
    print(f"Centroids shape: {centroids.shape}")
    return centroids, conv_dict


def assign_based_on_centroids(df, embedding_col_name, centroids):
    """
    Assigns each row in the df to the centroid closest to "embedding_col_name".
    """
    df = df.copy()
    vecs = np.array(df[embedding_col_name].tolist())
    similarities = cosine_similarity(vecs, centroids) 
    df['closest_centroid'] = similarities.argmax(axis=1)
    df['closest_centroid_similarity'] = similarities.max(axis=1)
    return df



def alg_for_par_selection_with_initial_cents(df_with_text_list_embeds_initial_assignments, centroids, initial_assignment_col_name, text_list_embed_col_name, text_list_col_name='pars', iter_lim=100, verbose=True):
    """ 
    This is our implementation of the paragraph selection algorithm, used during inference.
    """
    df = df_with_text_list_embeds_initial_assignments.copy()
    cat_d = {k:i for i, k in enumerate(df['category'].unique())}
    df['cat_idx'] = df['category'].apply(lambda x: cat_d[x])
    sils = []
    df['predicted_centroid_idx'] = df[initial_assignment_col_name]
    df['predicted_centroid'] = df['predicted_centroid_idx'].apply(lambda x: centroids[x])
    df['sim_to_cent'] = df.apply(lambda x: cosine_similarity([x['predicted_centroid']], x[text_list_embed_col_name]),axis=1)
    df['closest_par_idx'] = df['sim_to_cent'].apply(lambda x: np.argmax(x))
    df['closest_par_txt'] = df.apply(lambda x: x[text_list_embed_col_name][x['closest_par_idx']], axis=1)
    df['closest_par'] = df.apply(lambda x: x[text_list_embed_col_name][x['closest_par_idx']], axis=1)
    prev_cents = centroids.copy()
    prev_cents = [c / np.linalg.norm(c) for c in prev_cents]  # normalize centroids
    prev_cents = np.array(prev_cents)
    for i in tqdm(range(iter_lim)) if verbose else range(iter_lim):
        # sils.append(silhouette_score(np.array(df['closest_par'].tolist()), df['cat_idx'].to_list(), metric='cosine'))
        sils.append(calinski_harabasz_score(np.array(df['closest_par'].tolist()), df['cat_idx'].to_list()))
        # find new centroids based on closest paragraphs:
        grouped_by_prev_cents = df.groupby('predicted_centroid_idx')
        prev_idx_to_new_cts_conv = {}
        new_centroids = []
        for g_name, g_df in grouped_by_prev_cents:
            closest_pars = np.array(g_df['closest_par'].to_list())
            new_centroid = closest_pars.mean(axis=0)
            new_centroid = new_centroid / np.linalg.norm(new_centroid)
            new_centroids.append(new_centroid)
            prev_idx_to_new_cts_conv[g_name] = new_centroid
        new_centroids = np.array(new_centroids)
        # find similarity between new centroids and old centroids
        sim = cosine_similarity(prev_cents, new_centroids)
        matrix_of_1s = np.ones(sim.shape)
        if list(np.diag(sim)) == list(np.diag(matrix_of_1s)):
            print(f"Converged after {i} iterations due to no change in centroids")
            break
        prev_cents = new_centroids
        df['new_centroid'] = df['predicted_centroid_idx'].apply(lambda x: prev_idx_to_new_cts_conv[x])
        df['new_sim_to_cent'] = df.apply(lambda x: cosine_similarity([x['new_centroid']], x[text_list_embed_col_name])[0], axis=1)
        df['new_closest_par_idx'] = df['new_sim_to_cent'].apply(lambda x: np.argmax(x))
        df['new_closest_par'] = df.apply(lambda x: x[text_list_embed_col_name][x['new_closest_par_idx']], axis=1)
        df['new_closest_par_txt'] = df.apply(lambda x: x[text_list_col_name][x['new_closest_par_idx']], axis=1)
        if df['new_closest_par_idx'].to_list() == df['closest_par_idx'].to_list():
            if verbose:
                print(f"Converged after {i} iterations due to no change in closest paragraphs")
            break
        df['closest_par_idx'] = df['new_closest_par_idx']
        df['closest_par'] = df['new_closest_par']
        df['closest_par_txt'] = df['new_closest_par_txt']
    if i == iter_lim:
        print(f"Reached iteration limit of {iter_lim} without convergence.")
    df['new_closest_par_vote'] = df['new_closest_par'].apply(lambda x: np.argmax(cosine_similarity([x], centroids)[0]))
    df['closest_par_wrong'] = df.apply(lambda x: 1 if x['new_closest_par_vote'] != x['predicted_centroid_idx'] else 0, axis=1)
    s = df['closest_par_wrong'].sum()
    if s > 0 and verbose:
        print(f"Warning: {s} paragraphs were selected that vote differently than the predicted centroid.")
    if verbose:
        print(f"Silhouette scores during iterations\n {sils}")
    return df
    

def alg_for_par_selection(df_with_text_list_embeds, centroids, initial_rep_col_name, text_list_embed_col_name,text_list_col_name='pars', iter_lim = 100, verbose=True):
    """"
    This version of the algorithm is used for the train set, where the initial assignments are trivial.
    """
    df = df_with_text_list_embeds.copy()
    cat_d = {k:i for i, k in enumerate(df['category'].unique())}
    df['cat_idx'] = df['category'].apply(lambda x: cat_d[x])
    sils = []
    df['sim_to_cents'] = df[initial_rep_col_name].apply(lambda x: cosine_similarity([x], centroids)[0])
    df['predicted_centroid_idx'] = df['sim_to_cents'].apply(lambda x: np.argmax(x))
    sils.append(silhouette_score(np.array(df[initial_rep_col_name].tolist()), df['cat_idx'].to_list(), metric='cosine'))
    df['predicted_centroid'] = df['predicted_centroid_idx'].apply(lambda x: centroids[x])
    df['sim_to_cent'] = df.apply(lambda x: cosine_similarity([x['predicted_centroid']], x[text_list_embed_col_name]),axis=1)
    df['closest_par_idx'] = df['sim_to_cent'].apply(lambda x: np.argmax(x))
    df['closest_par_txt'] = df.apply(lambda x: x[text_list_embed_col_name][x['closest_par_idx']], axis=1)
    df['closest_par'] = df.apply(lambda x: x[text_list_embed_col_name][x['closest_par_idx']], axis=1)
    prev_cents = centroids.copy()
    prev_cents = [c / np.linalg.norm(c) for c in prev_cents]  # normalize centroids
    prev_cents = np.array(prev_cents)
    sils = []
    for i in tqdm(range(iter_lim)):
        sils.append(silhouette_score(np.array(df['closest_par'].tolist()), df['cat_idx'].to_list(), metric='cosine'))
        # find new centroids based on closest paragraphs:
        grouped_by_prev_cents = df.groupby('predicted_centroid_idx')
        prev_idx_to_new_cts_conv = {}
        new_centroids = []
        for g_name, g_df in grouped_by_prev_cents:
            closest_pars = np.array(g_df['closest_par'].to_list())
            new_centroid = closest_pars.mean(axis=0)
            new_centroid = new_centroid / np.linalg.norm(new_centroid)
            new_centroids.append(new_centroid)
            prev_idx_to_new_cts_conv[g_name] = new_centroid
        new_centroids = np.array(new_centroids)
        # find similarity between new centroids and old centroids
        sim = cosine_similarity(prev_cents, new_centroids)
        matrix_of_1s = np.ones(sim.shape)
        if list(np.diag(sim)) == list(np.diag(matrix_of_1s)):
            print(f"Converged after {i} iterations due to no change in centroids")
            break
        prev_cents = new_centroids
        df['new_centroid'] = df['predicted_centroid_idx'].apply(lambda x: prev_idx_to_new_cts_conv[x])
        df['new_sim_to_cent'] = df.apply(lambda x: cosine_similarity([x['new_centroid']], x[text_list_embed_col_name])[0], axis=1)
        df['new_closest_par_idx'] = df['new_sim_to_cent'].apply(lambda x: np.argmax(x))
        df['new_closest_par'] = df.apply(lambda x: x[text_list_embed_col_name][x['new_closest_par_idx']], axis=1)
        df['new_closest_par_txt'] = df.apply(lambda x: x[text_list_col_name][x['new_closest_par_idx']], axis=1)
        if df['new_closest_par_idx'].to_list() == df['closest_par_idx'].to_list():
            if verbose:
                print(f"Converged after {i} iterations due to no change in closest paragraphs")
            break
        df['closest_par_idx'] = df['new_closest_par_idx']
        df['closest_par'] = df['new_closest_par']
        df['closest_par_txt'] = df['new_closest_par_txt']
    if i == iter_lim:
        print(f"Reached iteration limit of {iter_lim} without convergence.")
    df['new_closest_par_vote'] = df['new_closest_par'].apply(lambda x: np.argmax(cosine_similarity([x], centroids)[0]))
    df['closest_par_wrong'] = df.apply(lambda x: 1 if x['new_closest_par_vote'] != x['predicted_centroid_idx'] else 0, axis=1)
    s = df['closest_par_wrong'].sum()
    if s > 0:
        print(f"Warning: {s} paragraphs were selected that vote differently than the predicted centroid.")
    if verbose:
        print(f"Silhouette scores during iterations\n {sils}")
    return df





def get_embed_votes_and_f1(df, text_list_embed_col, centroids, selection_criterion, k):
    """"
    Implementation of the voter set selection step.
    """
    vote_df = df.copy()
    vote_df['sim_to_cents'] = vote_df[text_list_embed_col].apply(lambda x: cosine_similarity(x, centroids))  
    vote_df['arg_highest_and_highest_per_par'] = vote_df['sim_to_cents'].apply(lambda x: (np.argmax(x, axis=1), np.max(x, axis=1)))
    if selection_criterion == 'first':
        vote_df['top_k_winner'] = vote_df['arg_highest_and_highest_per_par'].apply(lambda x: mode(x[0][1:k+1]))
    elif selection_criterion == 'highest_confidence':
        vote_df['top_k_winner'] = vote_df['arg_highest_and_highest_per_par'].apply(lambda x: mode([x[0][1]] + [x[0][i] for i in np.argsort(x[1])[::-1][:k-1]]))
    elif selection_criterion == 'highest_certainty':
        vote_df['entropies'] = vote_df['sim_to_cents'].apply(lambda x: [entropy(softmax(p_dist)) for p_dist in x])
        vote_df['arg_highest_entropies'] = vote_df['entropies'].apply(lambda x: np.argsort(x))
        vote_df['preds_ordered_by_entropy'] = vote_df.apply(lambda x: [x['arg_highest_and_highest_per_par'][0][i] for i in x['arg_highest_entropies']], axis=1)
        vote_df['top_k_winner'] = vote_df.apply(lambda x: mode([x['arg_highest_and_highest_per_par'][0][1]] + x['preds_ordered_by_entropy'][:k-1]),axis=1)
    elif selection_criterion == 'random':
        vote_df['top_k_winner'] = vote_df['arg_highest_and_highest_per_par'].apply(lambda x: mode(x[0][1] + [x[0][i] for i in random.sample(len(x[0]), k-1)]))
    else:
        raise ValueError(f"Unknown selection criterion: {selection_criterion}")
    pair_df = pd.merge(vote_df[['category', 'top_k_winner', 'id']], vote_df[['category', 'top_k_winner', 'id']], how='cross', suffixes=('_1', '_2'))
    pair_df['label'] = pair_df.apply(lambda x: 1 if x['category_1'] == x['category_2'] else 0, axis=1)
    pair_df['pred'] = pair_df.apply(lambda x: 1 if x['top_k_winner_1'] == x['top_k_winner_2'] else 0, axis=1)
    f1 = get_macro_f1(pair_df)[0]
    return vote_df, f1




def main():
    pass


if __name__ == "__main__":
    main()


