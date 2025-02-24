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
from collections import OrderedDict
from sklearn.metrics import mean_squared_error
import scipy.stats
import argparse
import wandb

def standardize(x):
    return (x - x.mean(axis=0))/(x.std(axis=0))
    
class FlabDataset(Dataset):
    def __init__(
        self, 
        csv_path, 
        target_col,
        split='train',
        train_test_split=0.8,
        split_seed=2023,
    ):
        super().__init__()

        data = pd.read_csv(csv_path, sep=",")

        train_df = data.sample(frac=train_test_split, random_state=split_seed)
        test_df = data.drop(train_df.index)

        if split=='train':
            self.heavy = train_df['heavy'].tolist()
            self.light = train_df['light'].tolist()
            self.target = standardize(train_df[target_col].to_numpy())
        if split=='test':
            test_df = test_df.sample(frac=1.0)
            self.heavy = test_df['heavy'].tolist()
            self.light = test_df['light'].tolist()
            self.target = standardize(test_df[target_col].to_numpy())

    def __len__(self):
        return len(self.heavy)

    def __getitem__(self, index):
        return self.heavy[index], self.light[index], self.target[index]

class FlabCollateFn:
    
    def __init__(self, truncation_seq_length=None):
        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        batch_size = len(batches)
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
        self.project = nn.Sequential(
            nn.Linear(1280, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, chains, chain_ids):
        chain_out = self.model(chains, chain_ids, repr_layers=[33])["representations"][33].mean(1)
        out = self.project(chain_out)
        return out


@torch.no_grad()
def evaluate(model, loader, args):

    device = args.device

    pred = []
    targets = []

    for step, eval_batch in enumerate(tqdm(loader)):
        
        chains, chain_ids, target = eval_batch
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)
        target = target.to(device)

        pred_ddg = model(chains, chain_ids)

        pred.append(pred_ddg.squeeze(-1).detach().cpu().numpy())
        targets.append(target.cpu().numpy())

    pred = np.concatenate(pred).ravel()
    targets = np.concatenate(targets).ravel()

    mse = mean_squared_error(pred, targets, squared=False)
    pearson =  scipy.stats.pearsonr(pred, targets)[0]
    spearman =  scipy.stats.spearmanr(pred, targets)[0]

    print(mse, pearson, spearman)

    return mse, pearson, spearman

def train(model, train_loader, val_loader, cfg, args):
    device = args.device
    
    optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr
        )
    loss_fn = torch.nn.MSELoss()
    model.to(device)

    for epoch in range(args.num_epochs):
        print(f'Training at epoch {epoch}')
        loss_accum = 0
        for step, train_batch in enumerate(tqdm(train_loader)):
            
            model.train()
            optimizer.zero_grad()

            chains, chain_ids, target = train_batch
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            target = target.to(device)

            pred_ddg = model(chains, chain_ids)
            loss = loss_fn(pred_ddg.squeeze(-1), target.float())

            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        print(f'Loss at end of epoch {epoch}: {loss_accum/(step+1)}')
        
        print(f'Evaluating at epoch {epoch}')
        mse, pearson, spearman = evaluate(model, val_loader, args)

        if args.wandb:
            wandb.log({"train_loss": loss_accum/(step+1),
                      "mse": mse,
                      "pearson": pearson,
                      "spearman": spearman})


def main(args):

    dataset_files = ['datasets/Koenig2017_g6_er.csv',
                     'datasets/Koenig2017_g6_Kd.csv',
                     'datasets/Warszawski2019_d44_Kd.csv']
    
    for dataset_file in dataset_files:
        if args.wandb:
            os.environ["WANDB_CONFIG_DIR"] = "./"
            os.environ["WANDB_CACHE_DIR"] = "./"
            wandb.login(key=args.wandb_key)
            wandb.init(
                entity='bergerlab-mit',
                project="esm-multimer-downstream",
                config=args,
                group="Flab",
                job_type=dataset_file
                
            )

        train_dataset = FlabDataset(dataset_file, 'fitness', split='train')
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, collate_fn=FlabCollateFn(), shuffle=True)

        test_dataset = FlabDataset(dataset_file, 'fitness', split='test')
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, collate_fn=FlabCollateFn(), shuffle=False
            )
    
        cfg = argparse.Namespace()
        with open(f"/data/cb/scratch/varun/esm-multimer/esm-multimer/models/esm2_t33_650M_UR50D.json") as f:
            cfg.__dict__.update(json.load(f))
            
        model = FlabWrapper(cfg, args.checkpoint_path, args.freeze_percent, args.use_multimer, args.device)
    
        train(model, train_loader, test_loader, cfg, args)

        if args.wandb:
            wandb.finish()

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finetuning on Flab dataset')
    
    # General args
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wandb', action="store_true", default=False)
    parser.add_argument('--wandb_key', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default="/data/cb/scratch/share/esm-multimer/650M_reinit_resume/epoch=0-step=65000.ckpt")
    parser.add_argument('--freeze_percent', type=float, default=1.0)
    parser.add_argument('--use_multimer', action="store_true", default=False)
    parser.add_argument('--device', type=str, default="cuda:0")    

    args = parser.parse_args()
    main(args)
