import re
import argparse, json
import random
import math
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
#from tqdm.notebook import tqdm

import plm_multimer
from plm_multimer.model.esm2 import ESM2
from collections import OrderedDict
import scipy.stats

import wandb

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer

import warnings
warnings.filterwarnings("ignore")

class FlabDataset(Dataset):
    def __init__(
        self, 
        csv_path, 
        target_col
    ):
        super().__init__()

        data = pd.read_csv(csv_path, sep=",")
        self.heavy = data['heavy'].tolist()
        self.light = data['light'].tolist()
        self.target = data[target_col].tolist()
        
    def __len__(self):
        return len(self.heavy)

    def __getitem__(self, index):
        return self.heavy[index], self.light[index], self.target[index]

class DesautelsCollateFn:
    
    def __init__(self, truncation_seq_length=None):
        self.alphabet = plm_multimer.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        len(batches)
        heavy_chain, light_chain, labels = zip(*batches)
        chains = [self.convert(c) for c in [heavy_chain, light_chain]]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        labels = torch.from_numpy(np.stack(labels, 0))
        return chains, chain_ids, labels
        
    def convert(self, seq_str_list):
        batch_size = len(seq_str_list)
        seq_encoded_list = [self.alphabet.encode('<cls>' + seq_str.replace('J', 'L') + '<eos>') for seq_str in seq_str_list]
        if self.truncation_seq_length:
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                    seq_encoded_list[i] = seq[start:start+self.truncation_seq_length]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        if self.truncation_seq_length:
            assert max_len <= self.truncation_seq_length
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)
        
        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i,:len(seq_encoded)] = seq
        return tokens
    
def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict

class FlabWrapper(nn.Module):
    
    def __init__(self, cfg, checkpoint_path, freeze_percent=0.0, use_multimer=True, device='cuda:0'):
        super().__init__()
        self.cfg = cfg
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer = use_multimer,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if use_multimer:
            # remove 'model.' in keys
            new_checkpoint = OrderedDict((key.replace('model.', ''), value) for key, value in checkpoint['state_dict'].items())
            self.model.load_state_dict(new_checkpoint)
        else:
            new_checkpoint = upgrade_state_dict(checkpoint['model'])
            self.model.load_state_dict(new_checkpoint)
        total_layers = 33
        for name, param in self.model.named_parameters():
            if 'embed_tokens.weight' in name or '_norm_after' in name or 'lm_head' in name:
                param.requires_grad = False
            else:
                layer_num = name.split('.')[1]
                if int(layer_num) <= math.floor(total_layers*freeze_percent):
                    param.requires_grad = False

    def forward(self, chains, chain_ids):
        mask = (~chains.eq(self.model.cls_idx)) & (~chains.eq(self.model.eos_idx)) & (~chains.eq(self.model.padding_idx))
        chain_out = self.model(chains, chain_ids, repr_layers=[33])["representations"][33]
        mask_expanded = mask.unsqueeze(-1).expand_as(chain_out)
        masked_chain_out = chain_out * mask_expanded
        sum_masked = masked_chain_out.sum(dim=1)
        mask_counts = mask.sum(dim=1, keepdim=True).float()  # Convert to float for division
        mean_chain_out = sum_masked / mask_counts
        return mean_chain_out


@torch.no_grad()
def get_embeddings(model, loader, device='cuda'):

    model.to(device)

    embeddings = []
    targets = []

    for step, eval_batch in enumerate(tqdm(loader)):
        
        chains, chain_ids, target = eval_batch
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)
        target = target.to(device).float()

        embedding = model(chains, chain_ids)    

        embeddings.append(embedding.detach().cpu().numpy())
        targets.append(target.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    targets = np.concatenate(targets)
    
    return embeddings, targets

def gaussian_transform(y):
    y = PowerTransformer().fit_transform(y.reshape(-1,1))
    return y
    
def cross_validate(embeddings, targets, scale_all=True):
    lambda_grid = np.logspace(0, -6, num=7).tolist()  # creates [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    lambda_grid.append(0)  # Append 0 to the list of lambdas
    param_grid = {'alpha': lambda_grid}
    
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=0)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=0)
    
    X_scaled = embeddings

    if scale_all:
        targets = gaussian_transform(targets)
    
    # Initialize the Ridge Regression model
    ridge_model = Ridge()
    
    # Setup the GridSearchCV object
    clf = GridSearchCV(estimator=ridge_model, param_grid=param_grid, cv=inner_cv, scoring='r2')

    outer_scores = []
    outer_corrs = []

    for train_idx, test_idx in tqdm(outer_cv.split(X_scaled), total=10):
        # Split data into training and test sets for the outer CV
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        Y_train, Y_test = targets[train_idx], targets[test_idx]

        if not scale_all:
            Y_train = gaussian_transform(Y_train)
            Y_test = gaussian_transform(Y_test)
        
        # Fit the model (and find the best lambda using inner CV)
        clf.fit(X_train, Y_train)
        
        # Best model found by GridSearchCV
        best_model = clf.best_estimator_
        
        # Evaluate the best model on the outer test set
        Y_pred = best_model.predict(X_test)
        r2 = r2_score(Y_test, Y_pred)
        corr = scipy.stats.pearsonr(Y_test[:,0], Y_pred[:,0])[0]

        # Append the score
        outer_scores.append(r2)
        outer_corrs.append(corr)

    # Output the performance
    # print("R2 for each fold:", outer_scores)
    print("Average R2 across all folds:", np.mean(outer_scores))
    print("R2 Standard deviation across all folds:", np.std(outer_scores))
    print("Average pearson correlation across all folds:", np.mean(outer_corrs))
    print("pearson correlation tandard deviation across all folds:", np.std(outer_corrs))
    print("\n")

    scores_dict = {'R2_avg': np.mean(outer_scores),
                  'R2_std': np.std(outer_scores),
                  'Pearson_avg': np.mean(outer_corrs),
                  'Pearson_std': np.std(outer_corrs)}
    
    return scores_dict


def calculate_scores(model, dataset_file, device):    
    print(f"Evaluating for dataset file {dataset_file}")

    if 'Kd' in dataset_file or 'kd' in dataset_file:
        dataset = FlabDataset(dataset_file, 'negative log Kd')
    else:
        dataset = FlabDataset(dataset_file, 'negative log expression')
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, collate_fn=DesautelsCollateFn(), shuffle=False
    )
    embeddings, targets = get_embeddings(model, loader, device=device)
    targets = np.array(dataset.target)
    scores_dict = cross_validate(embeddings, targets, scale_all=True)
    return scores_dict


def main(args):
    dataset_files = ['datasets/Shanehsazzadeh2023_trastuzumab_zero_kd_ppl.csv',
            'datasets/Warszawski2019_d44_Kd_ppl.csv',
            'datasets/Koenig2017_g6_Kd_ppl.csv',
            'datasets/Koenig2017_g6_er_ppl.csv']

    cfg = argparse.Namespace()
    with open(f"/data/cb/scratch/varun/esm-multimer/esm-multimer/models/esm2_t33_650M_UR50D.json") as f:
        cfg.__dict__.update(json.load(f))

    model = FlabWrapper(cfg, args.checkpoint_path, 1.0, args.use_multimer, args.device)
    
    for dataset_file in dataset_files:
        scores_dict = calculate_scores(model, dataset_file, args.device)
        wandb.log(scores_dict)
        wandb.finish()
                     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetuning on Flab dataset')
    
    parser.add_argument('--wandb', action="store_true", default=False)
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default="/data/cb/scratch/share/esm-multimer/650M_reinit_resume/epoch=0-step=65000.ckpt")
    parser.add_argument('--use_multimer', action="store_true", default=False)
    parser.add_argument('--device', type=str, default="cuda:0")    

    args = parser.parse_args()
    main(args)




