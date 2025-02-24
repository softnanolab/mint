import os,sys,re
import argparse, json
import copy
import random
import pickle
import math
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
#from tqdm.notebook import tqdm
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import one_to_index
from Bio.PDB import Selection
from Bio import SeqIO
from Bio.PDB.Residue import Residue
from easydict import EasyDict
import enum
sys.path.append('/data/cb/scratch/varun/esm-multimer/esm-multimer/')
import esm, gzip
from Bio import SeqIO
from esm.model.esm2 import ESM2
from esm.model.esm1 import ProteinBertModel
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, PredefinedSplit

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import wandb

class FlabDataset(Dataset):
    def __init__(
        self, 
        df,
        col1, 
        col2,
        col3,
        target_col,
        spike_only
    ):
        super().__init__()

        self.df = df
        self.col1 = col1
        self.col2 = col2
        self.col3 = col3
        self.target_col = target_col
        self.spike_only = spike_only
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        ab_1 = row[self.col1]
        ab_2 = row[self.col2]
        ag = row[self.col3]
        target = row[self.target_col]

        if self.spike_only:
            ag = ag[300:600]

        return ab_1, ab_2, ag, target

class DesautelsCollateFn:
    
    def __init__(self, truncation_seq_length=None, cat_chains=True):
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length
        self.cat_chains = cat_chains

    def __call__(self, batches):
        batch_size = len(batches)

        batch_inputs = list(zip(*batches))
        
        labels = batch_inputs[-1]
        chains = [self.convert(c) for c in batch_inputs[:-1]]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]

        labels =  torch.from_numpy(np.stack(labels, 0))
        
        if self.cat_chains:
            chains = torch.cat(chains, -1)
            chain_ids = torch.cat(chain_ids, -1)
            return chains, chain_ids, labels
        else:
            return chains[0], chains[1], labels

        
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


def convert_train_test_labels(train, test):
    t = PowerTransformer()
    train = t.fit_transform(train.reshape(-1,1))
    test = t.transform(test.reshape(-1,1))
    return train[:,0], test[:,0]

def classification_metrics(targets, predictions, threshold=0.5):
    binary_predictions = (predictions >= threshold).astype(int)
    accuracy = accuracy_score(targets, binary_predictions)
    f1 = f1_score(targets, binary_predictions)
    auc_score = roc_auc_score(targets, predictions)
    precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions)
    auprc = auc(recall_vals, precision_vals)
    return {
        'Accuracy': accuracy,
        'AUPRC': auprc,
        'F1 Score': f1,
        'AUROC': auc_score,
    }

def convert_train_test_features(train, test):
    t = StandardScaler()
    train = t.fit_transform(train)
    test = t.transform(test)
    return train, test

cfg = argparse.Namespace()
with open(f"/data/cb/scratch/varun/esm-multimer/esm-multimer/models/esm2_t33_650M_UR50D.json") as f:
    cfg.__dict__.update(json.load(f))

def calculate_scores(args):

    model = FlabWrapper(cfg, args.checkpoint_path, 1.0, args.use_multimer, args.device)

    os.environ["WANDB_CONFIG_DIR"] = "./"
    os.environ["WANDB_CACHE_DIR"] = "./"
    wandb.login(key='ecb8a6f984ef9af94ad2f544b82d7a91adc50dd5')
    wandb.init(
        entity='bergerlab-mit',
        project="esm-multimer-specific",
        group="covid",
        job_type=args.name,
    )

    train_df = pd.read_csv('processed_data_train.csv')
    test_df = pd.read_csv('processed_data_test.csv')

    train_dataset = FlabDataset(train_df, 'heavy', 'light', 'covid_seq', 'target', args.spike_only)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, collate_fn=DesautelsCollateFn(), 
        shuffle=True
    )
    train_embeddings, train_targets = get_embeddings(model, train_loader, args.device)
    
    test_dataset = FlabDataset(test_df, 'heavy', 'light', 'covid_seq', 'target', args.spike_only)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.bs, collate_fn=DesautelsCollateFn(), 
        shuffle=False
    )
    test_embeddings, test_targets = get_embeddings(model, test_loader, args.device)
    
    train_embeddings, test_embeddings = convert_train_test_features(train_embeddings, test_embeddings)
    
    verbose = 10
    n_jobs = -1
    model = MLPClassifier()
    param_grid = {
            "activation": ["relu"],
            "alpha": [0.0001],
            "learning_rate": ["adaptive"],
            "solver": ["adam"],
            "learning_rate_init": [0.001],
            "max_iter": [100],
            "hidden_layer_sizes": [
                (1280 // 2,),
                ],
            "early_stopping": [True],
            "random_state": [0],
            "validation_fraction": [0.1],
            "tol": [1e-4]}
    
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=verbose, scoring='roc_auc')

    grid_search.fit(train_embeddings, train_targets)
    
    # Best model found by GridSearchCV
    best_model = grid_search.best_estimator_
        
    # Evaluate the best model on the outer test set
    Y_pred = best_model.predict_proba(test_embeddings)[:,1]
    metrics = classification_metrics(test_targets, Y_pred)

    fl_name = 'best_embs_s1.npy' if args.spike_only else 'best_embs_full.npy'
    np.save(fl_name, Y_pred)

    wandb.log(metrics)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    # General args
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--bs', type=int, default=6)
    parser.add_argument('--use_multimer', action="store_true", default=False)
    parser.add_argument('--device', type=str, default="cuda:0")  
    parser.add_argument('--name', type=str, default="test")  
    parser.add_argument('--esm_type', type=str, default="esm2")  
    parser.add_argument('--spike_only', action="store_true", default=False)  

    args = parser.parse_args()
    calculate_scores(args)  

# python train.py --use_multimer --device "cuda:0" --checkpoint_path "/data/cb/scratch/varun/esm-multimer/esm-multimer/checkpoints/650M_nofreeze_filtered_continue/epoch=0-step=140000.ckpt" --name "esm-m-140" --spike_only