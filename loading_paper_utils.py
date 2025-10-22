import pandas as pd
import os
from tqdm import tqdm
from html_parsing_utils import get_content, extract_title_from_html, extract_abstract_from_html, extract_all_paragraphs_from_html, extract_all_paragraphs_from_html_v2
tqdm.pandas()
import numpy as np
from sentence_transformers import SentenceTransformer
import torch 
import torch.nn.functional as F

def load_data_from_pkl(path_to_data_pkl):
    """
    path_to_data_pkl: Path to a pkl file containing the data.
    """
    df = pd.read_pickle(path_to_data_pkl)
    if df is None:
        raise ValueError(f"File {path_to_data_pkl} does not exist.")
    if df.empty:
        raise ValueError(f"File {path_to_data_pkl} is empty.")
    return df


def embed_col(df, model,col_name, batch_size=50, model_name='', prefix=None, normalize=False):
    """
    adds a column of the vectors of the embedded titles.
    """
    all_embeds = []
    # if prefix:
    #     df[col_name] = df[col_name].apply(lambda x: prefix + x)
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_embeddings = model.encode([prefix+x if prefix else x for x in batch[col_name].tolist()], show_progress_bar=False)
        all_embeds.append(batch_embeddings)
    all_embeds = np.vstack(all_embeds)
    all_embeds = [np.array(x) for x in all_embeds]
    if normalize:
        all_embeds = [x / np.linalg.norm(x) for x in all_embeds]  # Normalize embeddings
    df[col_name +(f'_{model_name}' if model_name != '' else '')+'_embedding'] = all_embeds
    return df

def mean_pooling(last_hidden_state, attention_mask):
    # Apply mask to zero out padding tokens
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask



def embed_col_transformer(df, tokenizer, embed_model, col_name, batch_size=50, model_name='', prefix=None):
    all_embeds = []
    embed_model.eval()  # Set model to evaluation mode
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        batch_tokens = tokenizer([prefix+x if prefix else x for x in batch[col_name].tolist()], padding=True, truncation=True, return_tensors='pt').to(embed_model.device)
        outputs = embed_model(**batch_tokens)
        embeds = mean_pooling(outputs.last_hidden_state, batch_tokens['attention_mask'])
        batch_embeddings = F.normalize(embeds, p=2, dim=1).detach().cpu().numpy()  # Normalize embeddings
        all_embeds.append(batch_embeddings)
    all_embeds = np.vstack(all_embeds)
    all_embeds = [np.array(x) for x in all_embeds]
    df[col_name +(f'_{model_name}' if model_name != '' else '')+'_embedding'] = all_embeds
    return df

def embed_multi_col(df, model, column_name_list, batch_size=250, model_name='', prefix=None):
    """
    adds a column of the vectors of embeddings of concatenated data in the column_name_list.
    """
    all_embeds = []
    df[f'concatenated_{"_".join(column_name_list)}'] = df[column_name_list[0]].copy()
    for col in column_name_list[1:]:
        df[f'concatenated_{"_".join(column_name_list)}'] += ' ' + df[col]
    if prefix:
        df[f'concatenated_{"_".join(column_name_list)}'] = df[f'concatenated_{"_".join(column_name_list)}'].apply(lambda x: prefix + x)
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        to_encode = batch[f'concatenated_{"_".join(column_name_list)}']
        batch_embeddings = model.encode((to_encode).tolist(), show_progress_bar=False)
        all_embeds.append(batch_embeddings)
    all_embeds = np.vstack(all_embeds)
    all_embeds = [np.array(x) for x in all_embeds]
    df[f'concatenated_{"_".join(column_name_list)}'+(f'_{model_name}' if model_name != '' else '')+'_embedding'] = all_embeds
    return df



def embed_text_list_col(df, model, col_name, model_name='', prefix=None, normalize=False):
    """
    col_name has to be a column where the values are lists of texts.
    """
    embeds = []
    for i, r in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {col_name}"):
        row_embeds = model.encode([prefix +v if prefix else v for v in r[col_name]], show_progress_bar=False)
        if normalize:
            row_embeds = [x / np.linalg.norm(x) for x in row_embeds]  # Normalize embeddings
        embeds.append(row_embeds)
    df[col_name +(f'_{model_name}' if model_name != '' else '')+'_embedding'] = embeds
    return df

def embed_text_list_col_transformer(df, tokenizer, embed_model, col_name, model_name='', prefix=None):
    """
    col_name has to be a column where the values are lists of texts.
    """
    embeds = []
    embed_model.eval()  # Set model to evaluation mode
    for i, r in tqdm(df.iterrows(), total=len(df), desc=f"Embedding {col_name}"):
        texts = [prefix + v if prefix else v for v in r[col_name]]
        b_embeds = []
        for i in range(0, len(texts), 50):
            batch_texts = texts[i:i+50]
            batch_tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(embed_model.device)
            outputs = embed_model(**batch_tokens)
            row_embeds = mean_pooling(outputs.last_hidden_state, batch_tokens['attention_mask'])
            b_embeds.append(F.normalize(row_embeds, p=2, dim=1).detach().cpu().numpy())
            del row_embeds
            torch.cuda.empty_cache()  # Clear GPU memory
            # print(b_embeds[-1].shape)
        total_embeds = []
        for b in b_embeds:
            total_embeds = total_embeds + list(b)
        embeds.append(total_embeds)
    df[col_name +(f'_{model_name}' if model_name != '' else '')+'_embedding'] = embeds
    return df

def load_data_from_html_dir(html_dir):
    """
    Returns a pandas dataframe containing the titles, abstracts and categories of the research papers.
    html_dir: Path do a directory containing a folder for each category, each containing the paper html files.
    """
    if not os.path.exists(html_dir):
        raise ValueError(f"Directory {html_dir} does not exist.")
    data = []
    for category in os.listdir(html_dir):
        category_path = os.path.join(html_dir, category)
        if os.path.isdir(category_path):
            for filename in tqdm(os.listdir(category_path), desc=f"Processing {category}"):
                if filename.endswith(".html"):
                    file_path = os.path.join(category_path, filename)
                    html_content = get_content(file_path)
                    title = extract_title_from_html(html_content)
                    abstract = extract_abstract_from_html(html_content)
                    data.append({"id": filename.split('.html')[0], "title": title, "abstract": abstract, "category": category})
    df = pd.DataFrame(data)
    df = df[df.apply(lambda x: len(x['title']) > 0 and len(x['abstract']) > 0, axis=1)]
    return df


def add_path_to_html(df, main_htmls_dir):
    """
    df needs to have a column with the arxiv_id, and a column with the category.
    """
    df['html_path'] = df.apply(lambda x: os.path.join(main_htmls_dir, x['category'], x['id'] + '.html'), axis=1)
    if df['html_path'].isnull().any():
        raise ValueError("Some html paths are null. Please check the data.")
    if df['html_path'].apply(lambda x: not os.path.exists(x)).any():
        raise ValueError("Some html files do not exist. Please check the data.")


def add_paragraphs(df):
    # df has to have a path column
    paragraphs = []
    sections = []
    # print rows with empty paragraphs
    print(len(df))
    for i, r in tqdm(df.iterrows(), total=len(df), desc="Extracting paragraphs"):
        pars, secs = extract_all_paragraphs_from_html(get_content(r['html_path']))
        if pars is None or secs is None or len(pars) == 0 or len(secs) == 0:
            print(f'problem with {r["html_path"]}')
            paragraphs.append([])
            sections.append([])
        else:
            paragraphs.append(pars)
            sections.append(secs)
    print('here')
    df['pars'] = paragraphs
    df['sections'] = sections
    df['non_empty_par_indices'] = df['pars'].apply(lambda x: [i for i, p in enumerate(x) if len(p) > 0])
    df['pars'] = df.apply(lambda x: [x['pars'][i] for i in x['non_empty_par_indices']], axis=1)
    df['sections'] = df.apply(lambda x: [x['sections'][i] for i in x['non_empty_par_indices']], axis=1)
    df.drop(columns=['non_empty_par_indices'], inplace=True)
    indices_to_drop = df[df['pars'].apply(lambda x: len(x) == 0)].index
    df.drop(indices_to_drop, inplace=True)
    print(f'total of {len(df)} remained')
    df['num_pars'] = df['pars'].apply(lambda x: len(x))
    return df


def mean_embedding(df, col_name = 'pars_embedding'):
    """
    dataframe has to have a column with list of embeddings, this method adds the mean embedding of the list.
    """
    df[f'mean_{col_name.replace("embedding", "")}_embedding'] = df[col_name].progress_apply(lambda x: np.vstack(x).mean(axis=0))


def main():
    # example usage
    return
if __name__ == '__main__':
    # example usage
    df = pd.read_csv('YOUR/DATA/CSV/PATH')
    add_path_to_html(df, 'PATH/TO/DIR/WITH/HTML/PER/CATEGORY/ARXIV_ID')
    df = add_paragraphs(df)
    print(df.columns)
    df.to_pickle('SAVE/PATH/TO/PKL/FILE')
