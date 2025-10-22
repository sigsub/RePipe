import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torch
from sentence_transformers import SentenceTransformer
from loading_paper_utils import load_data_from_pkl
import os
import matplotlib.pyplot as plt
from loading_paper_utils import load_data_from_pkl, embed_col, embed_text_list_col
import numpy as np
import random
from pytorch_metric_learning.losses import ContrastiveLoss
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set to the GPU you want to use, e.g., "0" for the first and "1" for the second.


def text_list_embedded_df_to_data_pairs(path_to_df_pkl, text_list_col_embeds_name, underlying_property_col_name, save_dir):
    df = load_data_from_pkl(path_to_df_pkl)
    os.makedirs(save_dir, exist_ok=True)
    df.reset_index(drop=True, inplace=True)
    print(df.columns)
    vec_shape = df[text_list_col_embeds_name].iloc[0][0].shape[0]
    max_len = df[text_list_col_embeds_name].apply(lambda x: len(x)).max()
    df[text_list_col_embeds_name] = df[text_list_col_embeds_name].apply(lambda x: torch.tensor(x))
    # print(df[text_list_col_embeds_name].iloc[0].shape[0])
    df['padding_indicator'] = df[text_list_col_embeds_name].apply(lambda x: torch.cat((torch.zeros(x.shape[0]), torch.ones(max_len - x.shape[0]))))
    df[text_list_col_embeds_name] = df[text_list_col_embeds_name].apply(lambda x: torch.cat((x, torch.zeros((max_len - x.shape[0], vec_shape))), dim=0))
    pairs = []
    grouped = df.groupby(underlying_property_col_name).groups
    keys = list(grouped.keys())
    curr_i = 0
    for k in tqdm(keys):
        indices = grouped[k]
        print(f"Processing key {k} with {len(indices)} indices")
        l = len(indices)
        # print(f"total should be: {2*l*(l-1)} pairs")
        for i in range(len(indices)):
            # sample at most 100 positive indices in the range(i+1, len(indices))
            indices_to_consider = random.sample(list(range(i+1, len(indices))), min(25, len(indices) - i - 1))
            # for j in range(i+1, len(indices)):
            for j in indices_to_consider:
                idx1 = indices[i]
                idx2 = indices[j]
                vecs1 = df[text_list_col_embeds_name].iloc[idx1]
                pad_1 = df['padding_indicator'].iloc[idx1]
                vecs2 = df[text_list_col_embeds_name].iloc[idx2]
                pad_2 = df['padding_indicator'].iloc[idx2]
                pairs.append(((vecs1, pad_1), (vecs2, pad_2), 1))
                # sample 3 different keys
                different_keys = [t for t in random.sample(keys, len(keys)) if t != k][:3]
                for dk in different_keys:
                    didx = random.choice(grouped[dk])
                    dvecs = df[text_list_col_embeds_name].iloc[didx]
                    dpad = df['padding_indicator'].iloc[didx]
                    pairs.append(((vecs1, pad_1), (dvecs, dpad), 0))
        if len(pairs) > 10000:
            print(f"Saving {len(pairs)} pairs to {save_dir}")
            for i, p in tqdm(enumerate(pairs)):
                torch.save(p, os.path.join(save_dir, f"pair_{curr_i + i}.pt"))
            curr_i += len(pairs)
            pairs = []


    


def collator(batch):
    """
    batch: list of tuples of the form ((vecs1, pad_1), (vecs2, pad_2), label)
    vecs1, vecs2: tensors of shape (seq_len, d_model)
    pad_1, pad_2: tensors of shape (seq_len,)
    label: int
    Returns four tensors each being the concatenation of the corresponding tensors in the batch.
    """

    vecs1 = torch.stack([b[0][0] for b in batch])  # shape (batch_size, seq_len, d_model)
    pad_1 = torch.stack([b[0][1] for b in batch])  # shape (batch_size, seq_len)
    vecs2 = torch.stack([b[1][0] for b in batch])  # shape (batch_size, seq_len, d_model)
    pad_2 = torch.stack([b[1][1] for b in batch])  # shape (batch_size, seq_len)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float)  # shape (batch_size,)
    return (vecs1, pad_1), (vecs2, pad_2), labels



class PaperVecPairsDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder_path):
        """
        data_folder_path: path to folder containing data files
        """
        self.data_folder = data_folder_path

    def __len__(self):
        return len(os.listdir(self.data_folder))
    
    def __getitem__(self, idx):
        return torch.load(os.path.join(self.data_folder, f"pair_{idx}.pt"))





class transformer_agg_net(nn.Module):
    def __init__(self, d_model, nhead, n_transformer_layers, dropout):
        super(transformer_agg_net, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.n_transformer_layers = n_transformer_layers
        self.dropout = dropout
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=n_transformer_layers
        )

    @staticmethod
    def mean_pooling(vec_batch, attention_mask):
        """
        vec_batch: Tensor of shape (batch_size, seq_len, d_model)
        attention_mask: Tensor of shape (batch_size, seq_len) with True for padded positions
        Performs mean pooling over non-padded vectors in a batch.
        should return a tensor of shape (batch_size, d_model)
        Masked positions HAVE TO CONTAIN ZERO VECTORS.
        """
        sums = torch.sum(vec_batch, dim=1)
        attention_mask = attention_mask.to(torch.bool)  # ensure attention_mask is boolean
        included = ~attention_mask
        counts = torch.sum(included, dim=1)
        # divide each vec in sums by its corresponding count
        counts = counts.unsqueeze(1)  # make it (batch_size, 1)
        return sums / counts


    def forward(self, x):
        """
        x: a tuple of the input tensor and the attention mask
        x[0]: Tensor of shape (batch_size, seq_len, d_model)
        x[1]: Attention mask of shape (batch_size, seq_len)
        """
        rep = self.encoder(x[0], src_key_padding_mask=x[1])
        agg_rep = transformer_agg_net.mean_pooling(rep, x[1])
        return agg_rep, rep
    

def train(model, dataset, n_epochs, batch_size, lr, margin=0):
    """
    model is an instance of transformer_agg_net
    kwargs have to include
    """
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=True) # dataset is a PaperVecPairsDataset
    criterion = nn.CosineEmbeddingLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses =[]
    for epoch in range(n_epochs):
        losses_for_batch_print = []
        for i, batch in tqdm(enumerate(loader), total=len(dataset) // batch_size):
            vecs1, pad_1 = batch[0]
            vecs2, pad_2 = batch[1]
            labels = batch[2]
            vecs1 = vecs1.to('cuda')
            pad_1 = pad_1.to('cuda')
            vecs2 = vecs2.to('cuda')
            pad_2 = pad_2.to('cuda')
            labels = labels.to('cuda')
            labels = torch.flatten(labels)  # ensure labels are flattened
            labels = 2*labels - 1
            # break
            optimizer.zero_grad()
            out1, rep1 = model((vecs1, pad_1))
            out2, rep2 = model((vecs2, pad_2))
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()
            losses_for_batch_print.append(loss.item())
            if i % 50 == 0:
                print(f"Batch {i}, Loss: {sum(losses_for_batch_print)/len(losses_for_batch_print)}")
                losses.append(sum(losses_for_batch_print)/len(losses_for_batch_print))
                losses_for_batch_print = []
        print(f"Epoch {epoch+1}/{n_epochs}, mean average Loss: {sum(losses)/len(losses)}")
    fig, ax = plt.subplots()
    ax.plot(losses, label=f'Epoch {epoch+1}')
    plt.savefig(f'loss.png')


def embed_and_save_paper_df(path_to_df_pkl, model_name_list, model_list, prefix_list, save_path, text_list_col_name = 'pars', add_title_abstract=True):
    assert len(model_name_list) == len(model_list) == len(prefix_list), "model_name_list, model_list and prefix_list must have the same length"
    df = load_data_from_pkl(path_to_df_pkl)
    if add_title_abstract:
        df['pars'] = df.apply(lambda x: [x['title']] + [x['abstract']] + x[text_list_col_name], axis=1)
    for i in range(len(model_list)):
        model = model_list[i]
        model_name = model_name_list[i]
        prefix = prefix_list[i]
        print(f"Embedding with model {model_name}...")
        embed_text_list_col(df, model, text_list_col_name, model_name, prefix)
    df.to_pickle(save_path)

def gen_datasets_for_transformer_agg(train_or_test='train'):
    """
    Generates a dataset which can be used for training/testing the transformer aggregation model.
    """
    assert train_or_test in ['train', 'test'], "train_or_test must be either 'train' or 'test'"
    e5_model = SentenceTransformer('intfloat/e5-base-unsupervised')
    mpnet_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    models = [e5_model, mpnet_model]
    model_names = ['pt_e5', 'pt_all_mpnet']
    prefixes = ['query: ', '']
    embed_and_save_paper_df(
        path_to_df_pkl=f"PATH/TO/{train_or_test}/DATA/FILE.pkl",
        model_name_list=model_names,
        model_list=models,
        prefix_list=prefixes,
        save_path=f"PATH/TO/SAVE{train_or_test}/EMBEDDED_DF.pkl",
        text_list_col_name='pars',
        add_title_abstract=True
    )

def get_pairs_for_transformer_agg():
    for c in ['pars_e5_embedding', 'pars_all_mpnet_embedding']:
        text_list_embedded_df_to_data_pairs(
            path_to_df_pkl='/PATH/TO/EMEDDED/DF/FILE.pkl',
            text_list_col_embeds_name= c,
            underlying_property_col_name='category',
            save_dir = f"/PATH/TO/SAVE/{c}_pairs"
        )
def main():
    dataset = PaperVecPairsDataset(data_folder_path="PATH/TO/PAIR/DATASET/FOLDER")
    model = transformer_agg_net(d_model=768, nhead=12, n_transformer_layers=2, dropout=0.1).to('cuda')
    train(model, dataset, n_epochs=10, batch_size=64, lr=1e-5, margin=0)
    # torch.save(model.state_dict(), "news_parade_agg_model_2_layers_on_ft_e5.pth")

if __name__ == "__main__":
    main()
        
