import re
import argparse, json
import random
import math
import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm
#from tqdm.notebook import tqdm

import plm_multimer
from plm_multimer.model.esm2 import ESM2
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc

import warnings
warnings.filterwarnings("ignore")

import wandb

class PPIDataset(Dataset):
    def __init__(
        self, 
        df, 
        beta_col, 
        ag_col, 
        target_col,
        use_extra_info,
        alpha_col,
        mhc_col
    ):
        super().__init__()
        self.data_df = df
        self.beta_col = beta_col
        self.ag_col = ag_col
        self.alpha_col = alpha_col
        self.mhc_col = mhc_col
        self.use_extra_info = use_extra_info
        
        self.target_col = target_col

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        return row[self.beta_col], row[self.ag_col], row[self.target_col]

class PPICollateFn:
    
    def __init__(self, truncation_seq_length=None):
        self.alphabet = plm_multimer.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        len(batches)
        ab_chain, ag_chain, labels = zip(*batches)
        
        chains = [self.convert(c) for c in [ab_chain, ag_chain]]
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
            nn.Linear(1280*2, 1280),
            nn.SiLU(),
            nn.Linear(1280, 1)
        )

    def get_one_chain(self, chain_out, mask_expanded, mask):
        masked_chain_out = chain_out * mask_expanded
        sum_masked = masked_chain_out.sum(dim=1)
        mask_counts = mask.sum(dim=1, keepdim=True).float()  # Convert to float for division
        mean_chain_out = sum_masked / mask_counts
        return mean_chain_out

    def forward(self, chains, chain_ids):
        mask = (~chains.eq(self.model.cls_idx)) & (~chains.eq(self.model.eos_idx)) & (~chains.eq(self.model.padding_idx))
        chain_out = self.model(chains, chain_ids, repr_layers=[33])["representations"][33]

        mask_chain_0 = (chain_ids.eq(0) & mask).unsqueeze(-1).expand_as(chain_out) 
        mask_chain_1 = (chain_ids.eq(1) & mask).unsqueeze(-1).expand_as(chain_out)

        mean_chain_out_0 = self.get_one_chain(chain_out, mask_chain_0, (chain_ids.eq(0) & mask))
        mean_chain_out_1 = self.get_one_chain(chain_out, mask_chain_1, (chain_ids.eq(1) & mask))
        
        hidden_concat = torch.cat((mean_chain_out_0, mean_chain_out_1), -1)
        return self.project(hidden_concat)


def classification_metrics(targets, predictions, name, threshold=0.5):
    # Convert probabilities to binary predictions based on a threshold
    binary_predictions = (predictions >= threshold).astype(int)
    
    # Calculate accuracy
    accuracy = accuracy_score(targets, binary_predictions)
    
    # Calculate precision, recall, and F1 score
    f1 = f1_score(targets, binary_predictions)

    auc_score = roc_auc_score(targets, predictions)
    
    # Calculate AUPRC
    precision_vals, recall_vals, _ = precision_recall_curve(targets, predictions)
    auprc = auc(recall_vals, precision_vals)

    return {
        f'{name}_Accuracy': accuracy,
        f'{name}_AUPRC': auprc,
        f'{name}_F1 Score': f1,
        f'{name}_AUROC': auc_score,
    }


@torch.no_grad()
def evaluate(model, loader, args):
    device = args.device
    preds = []
    targets = []
    for step, eval_batch in enumerate(tqdm(loader)):

        chains, chain_ids, target = eval_batch
        chains = chains.to(device)
        chain_ids = chain_ids.to(device)
        target = target.to(device)
        pred = torch.nn.functional.sigmoid(model(chains, chain_ids))
        preds.append(pred.squeeze(-1).detach().cpu().numpy())
        targets.append(target.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    targets = np.concatenate(targets).ravel()

    return preds, targets

def train(model, train_loader, val_loader, cfg, args, prefix):
    device = args.device
    
    optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            betas=json.loads(cfg.adam_betas), 
            eps=cfg.adam_eps,
            weight_decay=cfg.weight_decay
        )

    pos_weight = torch.tensor([args.pos_weight])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

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

            pred = model(chains, chain_ids)
            loss = loss_fn(pred.squeeze(-1), target.float())

            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
        print(f'Loss at end of epoch {epoch}: {loss_accum/(step+1)}')

        print(f'Evaluating at epoch {epoch}')
        preds, targets = evaluate(model, val_loader, args)

        metrics = classification_metrics(targets, preds, prefix)
        metrics['num_val'] = len(preds)
        metrics['train_loss'] = loss_accum/(step+1)

        if args.wandb:
            wandb.log(metrics)



cfg = argparse.Namespace()
with open(f"/data/cb/scratch/varun/esm-multimer/esm-multimer/models/esm2_t33_650M_UR50D.json") as f:
    cfg.__dict__.update(json.load(f))

def calculate_scores(args):

    for split_type in ['RN']:    
    

        model = FlabWrapper(cfg, args.checkpoint_path, args.freeze_percent, args.use_multimer, args.device)
        train_df = pd.read_csv(f'./processed_data/{split_type}/train_{args.rep}.csv')
        test_df = pd.read_csv(f'./processed_data/{split_type}/test_{args.rep}.csv')
        
        train_dataset = PPIDataset(train_df, 'cdr3.beta', 'antigen.epitope', 'Y', 
                                   args.use_extra_info, 'cdr3.alpha', 'mhc.seq')

        print(train_dataset[0])
        
        train_labels = torch.tensor(train_df['Y'].tolist())
        num_zeros = (train_labels == 0).sum().item()
        num_ones = (train_labels == 1).sum().item()
        weights = torch.tensor([num_zeros, num_ones])
        weights = 1/weights
        samples_weight = torch.tensor([weights[t] for t in train_labels.int()]).double()
        
        num_to_draw = 2*num_ones
        sampler = WeightedRandomSampler(samples_weight, num_to_draw, replacement=False)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.bs, collate_fn=PPICollateFn(), sampler=sampler
        )

        test_dataset = PPIDataset(test_df, 'cdr3.beta', 'antigen.epitope', 'Y', 
                                   args.use_extra_info, 'cdr3.alpha', 'mhc.seq')
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.bs, collate_fn=PPICollateFn(), shuffle=False
        )
        
        train(model, train_loader, test_loader, cfg, args, 'test')

        if args.wandb:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    # General args
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--use_multimer', action="store_true", default=False)
    parser.add_argument('--device', type=str, default="cuda:0")  
    parser.add_argument('--save_name', type=str, default="test")  
    parser.add_argument('--bs', type=int, default=48) 
    parser.add_argument('--use_extra_info', action="store_true", default=False)
    parser.add_argument('--wandb', action="store_true", default=False)
    parser.add_argument('--freeze_percent', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--rep', type=int, default=0)
    parser.add_argument('--pos_weight', type=float, default=2.0)

    args = parser.parse_args()
    calculate_scores(args)  




