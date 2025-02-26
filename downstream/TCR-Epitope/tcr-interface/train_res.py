import json
import math
import sys

sys.path.append(".")
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm

pl.seed_everything(0)
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from teim_utils import *
from teim_utils import (
    decoding_one_mat,
    get_scores_contact,
    get_scores_dist,
    load_config,
    load_data,
)
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

import plm_multimer
from plm_multimer.model.esm2 import ESM2


class ResLevelSystem(pl.LightningModule):
    def __init__(self, model, args, train_set, val_set, device_default, only_int=False):
        super().__init__()

        self.model = model
        self.args = args
        self.train_set = train_set
        self.val_set = val_set
        self.device_default = device_default

        self.best_val_auprc = 0
        self.best_val_samples = 0

        self.only_int = only_int

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.args.bs, shuffle=True, collate_fn=PPICollateFn()
        )

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=32, shuffle=False, collate_fn=PPICollateFn())

    def forward(self, chains, chain_ids, epi_aa):
        return self.model(chains, chain_ids, epi_aa)

    def minimum_step(self, batch, device=None):

        if device is None:
            chains, chain_ids, epi_aa, dist_mat, contact_mat, mask_mat, _ = batch
        else:
            chains, chain_ids, epi_aa, dist_mat, contact_mat, mask_mat, _ = batch
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            dist_mat = dist_mat.to(device)
            contact_mat = contact_mat.to(device)
            mask_mat = mask_mat.to(device)
            epi_aa = epi_aa.to(device)

        pred = self.forward(chains, chain_ids, epi_aa)
        res_pred = pred["reslevel_out"]
        loss = self.get_loss(res_pred, [dist_mat, contact_mat, mask_mat])
        return loss, res_pred, dist_mat, contact_mat, mask_mat

    def training_step(self, batch, batch_idx):
        self.train()
        loss_dict, pred, dist, contact, mask = self.minimum_step(batch)
        for key, value in loss_dict.items():
            self.log(f"train/{key}", value)
        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss_dict, pred, dist, contact, mask = self.minimum_step(batch)

        for key, value in loss_dict.items():
            self.log(f"val/{key}", value)
        return loss_dict["loss"]

    def training_epoch_end(self, training_step_outputs):

        ## validating metric
        loss, scores, scores_samples = self.evaluate_model(self.val_dataloader())
        self.log_dict(
            {
                "val/corr": scores[0],
                "val/mse": scores[1],
                "val/mape": scores[2],
                "val/auc": scores[3],
                "val/auprc": scores[4],
            }
        )

    def evaluate_model(
        self, data_loader=None,
    ):
        self.eval()
        loss_dict = {"loss": 0, "loss_dist": 0, "loss_contact": 0}
        pred, dist, contact, mask = [], [], [], []

        for i, batch in enumerate(data_loader):
            loss_this, pred_, dist_, contact_, mask_ = self.minimum_step(
                batch, self.device_default
            )
            for key, value in loss_this.items():
                loss_dict[key] += value
            pred.extend(pred_.detach().cpu().numpy().tolist())
            dist.extend(dist_.detach().cpu().numpy().tolist())
            contact.extend(contact_.detach().cpu().numpy().tolist())
            mask.extend(mask_.detach().cpu().numpy().tolist())
        for key, value in loss_dict.items():
            loss_dict[key] /= len(data_loader)

        scores, scores_samples = self.get_scores(pred, [dist, contact, mask])

        if scores[-1] > self.best_val_auprc:
            self.best_val_auprc = scores[-1]
            self.best_val_samples = scores_samples

        return loss_dict, scores, scores_samples

    def predict_valset(self):
        self.eval()
        pred = {}
        cdr3_list = []
        epi_list = []
        for batch in tqdm(self.val_dataloader()):
            pdb_chain = batch[-1]["pdb_chains"]
            cdr3 = batch[-1]["cdr3_seqs"]
            epi = batch[-1]["epi_seqs"]
            loss_, pred_, dist_, contact_, mask_ = self.minimum_step(batch, self.device_default)
            for i, pdb in enumerate(pdb_chain):
                value = pred_.detach().cpu().numpy()[i]
                pred[pdb] = decoding_one_mat(value, len(cdr3[i]), len(epi[i]))
                cdr3_list.append(cdr3[i])
                epi_list.append(epi[i])

        loss_dict, scores, scores_samples = self.evaluate_model(self.val_dataloader())

        return pred, cdr3_list, epi_list, scores_samples

    def get_loss(self, pred, labels):
        dist, contact, mask = labels
        loss_dist = F.mse_loss(pred[..., 0], dist, reduction="none")
        loss_dist = loss_dist * mask
        loss_dist = torch.sum(loss_dist) / torch.sum(mask)

        loss_bd = F.binary_cross_entropy(pred[..., 1], contact.float(), reduction="none")
        loss_bd = loss_bd * mask
        loss_bd = torch.sum(loss_bd) / torch.sum(mask)

        if self.only_int:
            loss = loss_bd
        else:
            loss = loss_dist + 1 * loss_bd

        return {"loss": loss, "loss_dist": loss_dist, "loss_contact": loss_bd}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": StepLR(optimizer, self.args.patience, gamma=self.args.decay),
        }

    def get_scores(self, pred, labels):
        dist, contact, mask = labels
        avg_metrics_dist, metrics_dist = get_scores_dist(
            np.array(dist), np.array(pred)[..., 0], np.array(mask)
        )
        avg_metrics_bd, metrics_bd = get_scores_contact(
            np.array(contact), np.array(pred)[..., 1], np.array(mask)
        )
        return avg_metrics_dist + avg_metrics_bd, metrics_dist + metrics_bd


def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


class PPICollateFn:
    def __init__(self, truncation_seq_length=None):
        self.alphabet = plm_multimer.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        len(batches)

        cdr3_list = [item["cdr3_seqs"] for item in batches]
        epi_list = [item["epi_seqs"] for item in batches]
        dist_mat_list = [item["dist_mat"] for item in batches]
        mask_mat_list = [item["mask_mat"] for item in batches]
        contact_mat_list = [item["contact_mat"] for item in batches]
        [item["pdb_chains"] for item in batches]
        epi_aa_list = [item["epi"] for item in batches]

        cdr3_enc = self.convert(cdr3_list, 20 + 2)
        epi_enc = self.convert(epi_list, 12 + 2)

        chains = [cdr3_enc, epi_enc]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        epi_aa = torch.from_numpy(np.stack(epi_aa_list, 0))

        dist_mat = torch.from_numpy(np.stack(dist_mat_list, 0))
        mask_mat = torch.from_numpy(np.stack(mask_mat_list, 0))
        contact_mat = torch.from_numpy(np.stack(contact_mat_list, 0))

        metadata = {
            "pdb_chains": [item["pdb_chains"] for item in batches],
            "cdr3_seqs": cdr3_list,
            "epi_seqs": epi_list,
        }

        return chains, chain_ids, epi_aa, dist_mat, contact_mat, mask_mat, metadata

    def convert(self, seq_str_list, max_len):
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode("<cls>" + seq_str.replace("J", "L") + "<eos>")
            for seq_str in seq_str_list
        ]
        if self.truncation_seq_length:
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                    seq_encoded_list[i] = seq[start : start + self.truncation_seq_length]
        if self.truncation_seq_length:
            assert max_len <= self.truncation_seq_length
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)

        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, : len(seq_encoded)] = seq
        return tokens


class ResNet(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, data):
        tmp_data = self.cnn(data)
        out = tmp_data + data
        return out


class FlabWrapper(nn.Module):
    def __init__(
        self,
        cfg,
        checkpoint_path,
        ae_model_cfg,
        freeze_percent=0.0,
        use_multimer=True,
        dim_hidden=256,
        dropout=0.2,
        device="cuda:0",
    ):
        super().__init__()
        self.layers_inter = 2
        dim_seqlevel = 256
        self.cfg = cfg
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            token_dropout=cfg.token_dropout,
            use_multimer=use_multimer,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.ae_model_cfg = ae_model_cfg

        # remove 'model.' in keys
        new_checkpoint = OrderedDict(
            (key.replace("model.", ""), value) for key, value in checkpoint["state_dict"].items()
        )
        self.model.load_state_dict(new_checkpoint)

        ## feature extractor
        self.seq_cdr3 = nn.Sequential(
            nn.Conv1d(1280, dim_hidden, 1,), nn.BatchNorm1d(dim_hidden), nn.ReLU(),
        )
        self.seq_epi = nn.Sequential(
            nn.Conv1d(1280, dim_hidden, 1,), nn.BatchNorm1d(dim_hidden), nn.ReLU(),
        )

        if self.ae_model_cfg.path != "":
            ae_model = AutoEncoder(self.ae_model_cfg.dim_hid, self.ae_model_cfg.len_epi)
            self.ae_encoder = load_model_from_ckpt(self.ae_model_cfg.path, ae_model)
            for param in self.ae_encoder.parameters():
                param.requires_grad = False
            self.ae_linear = nn.Linear(self.ae_model_cfg.dim_hid, dim_hidden, bias=False)
        else:
            self.ae_encoder = None

        if self.ae_encoder != None:
            self.inter_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                        nn.BatchNorm2d(dim_hidden),
                        nn.ReLU(),
                    ),
                    nn.ModuleList(
                        [  # second layer, this layer add the ae pretrained vector
                            ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                            nn.Sequential(nn.BatchNorm2d(dim_hidden), nn.ReLU(),),
                        ]
                    ),
                    *[  # more cnn layers
                        nn.Sequential(
                            ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                            nn.BatchNorm2d(dim_hidden),
                            nn.ReLU(),
                        )
                        for _ in range(self.layers_inter - 2)
                    ],
                ]
            )
        else:
            self.inter_layers = nn.ModuleList(
                [
                    nn.Sequential(
                        ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                        nn.BatchNorm2d(dim_hidden),
                        nn.ReLU(),
                    ),
                    *[  # more cnn layers
                        nn.Sequential(
                            ResNet(nn.Conv2d(dim_hidden, dim_hidden, kernel_size=3, padding=1)),
                            nn.BatchNorm2d(dim_hidden),
                            nn.ReLU(),
                        )
                        for _ in range(self.layers_inter - 1)
                    ],
                ]
            )

        ## seq-level prediction
        self.seqlevel_outlyer = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(dim_seqlevel, 1),
            nn.Sigmoid(),
        )

        ## res-level prediction
        self.reslevel_outlyer = nn.Conv2d(
            in_channels=dim_hidden,
            out_channels=2,
            kernel_size=2 * self.layers_inter + 1,
            padding=self.layers_inter,
        )

    def forward(self, chains, chain_ids, epi_aa):
        (~chains.eq(self.model.cls_idx)) & (~chains.eq(self.model.eos_idx)) & (
            ~chains.eq(self.model.padding_idx)
        )
        chain_out = self.model(chains, chain_ids, repr_layers=[33])["representations"][33]

        len_epi = 12
        len_cdr3 = 20

        # remove start and end tokens
        cdr3_emb = chain_out[:, 1:21, :]
        epi_emb = chain_out[:, 23:35, :]

        cdr3_feat = self.seq_cdr3(cdr3_emb.transpose(1, 2))  # batch_size, dim_hidden, seq_len
        epi_feat = self.seq_epi(epi_emb.transpose(1, 2))

        if self.ae_encoder is not None:
            ae_feat = self.ae_encoder(epi_aa, latent_only=True)  # batch_size, dim_ae
            ae_feat = self.ae_linear(ae_feat)  # batch_size, dim_ae

        cdr3_feat_mat = cdr3_feat.unsqueeze(3).repeat(
            [1, 1, 1, len_epi]
        )  # batch_size, dim_hidden, len_cdr3, len_epi
        epi_feat_mat = epi_feat.unsqueeze(2).repeat(
            [1, 1, len_cdr3, 1]
        )  # batch_size, dim_hidden, len_cdr3, len_epi

        inter_map = cdr3_feat_mat * epi_feat_mat

        ## inter layers features
        for i in range(self.layers_inter):
            if (i == 1) and ((self.ae_encoder is not None)):  # add ae features
                if self.ae_encoder is not None:
                    vec = ae_feat.unsqueeze(2).unsqueeze(3)
                    inter_map = self.inter_layers[i][0](inter_map)
                    inter_map = inter_map + vec
                inter_map = self.inter_layers[i][1](inter_map)
            else:
                inter_map = self.inter_layers[i](inter_map)

        ## output layers
        # seq-level prediction
        seqlevel_out = self.seqlevel_outlyer(inter_map)
        # res-level prediction
        reslevel_out = self.reslevel_outlyer(inter_map)
        out_dist = torch.relu(reslevel_out[:, 0, :, :])
        out_bd = torch.sigmoid(reslevel_out[:, 1, :, :])
        reslevel_out = torch.cat([out_dist.unsqueeze(-1), out_bd.unsqueeze(-1)], axis=-1)

        return {
            "seqlevel_out": seqlevel_out,
            "reslevel_out": reslevel_out,
            "inter_map": inter_map,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./TEIM/train_teim/configs/reslevel_bothnew.yml"
    )
    parser.add_argument("--freeze_percent", type=float, default=0.95)
    parser.add_argument("--full_checkpoint_path", type=str, default="")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/data/cb/scratch/varun/esm-multimer/esm-multimer/checkpoints/650M_nofreeze_filtered_continue/epoch=0-step=180000.ckpt",
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bs", type=int, default=48)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--decay", type=float, default=0.5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--only_int", action="store_true", default=False)
    parser.add_argument("--only_split", type=int, default=9)

    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    split = config_path.split("/")[-1].split(".")[0]

    config.data.path.summary = "./TEIM/data/stcrdab_pdb.csv"
    config.data.path.mat = "./TEIM/data/contact_map"
    config.training.epochs = 150
    config.model.ae_model.path = ""

    datasets = load_data(config.data)
    train_set, val_set = datasets["train"], datasets["val"]

    for i_split, (train_set_this, val_set_this) in enumerate(zip(train_set, val_set)):

        if args.only_split != 9:
            if i_split != args.only_split:
                continue

        wandb_logger = WandbLogger(log_model=False)

        device_str = "cuda:" + str(args.device)

        cfg = argparse.Namespace()
        with open(
            f"/data/cb/scratch/varun/esm-multimer/esm-multimer/models/esm2_t33_650M_UR50D.json"
        ) as f:
            cfg.__dict__.update(json.load(f))
        esm_model = FlabWrapper(
            cfg, args.checkpoint_path, config.model.ae_model, 0, True, 256, 0.2, device_str
        )

        if args.full_checkpoint_path != "":
            checkpoint = torch.load(args.full_checkpoint_path, map_location=device_str)
            new_checkpoint = OrderedDict(
                (key.replace("model.", "", 1), value)
                for key, value in checkpoint["state_dict"].items()
            )
            esm_model.load_state_dict(new_checkpoint)

        total_layers = 33
        for name, param in esm_model.named_parameters():
            if "embed_tokens.weight" in name or "_norm_after" in name or "lm_head" in name:
                param.requires_grad = False
            elif "model.layers" in name:
                layer_num = name.split(".")[2]
                if int(layer_num) <= math.floor(total_layers * args.freeze_percent):
                    param.requires_grad = False

        print("Split {}".format(i_split), "Train:", len(train_set_this), "Val:", len(val_set_this))
        # load model and trainer
        print("Loading model and trainer...")

        model = ResLevelSystem(
            esm_model, args, train_set_this, val_set_this, device_str, args.only_int
        )

        save_path = os.path.join(os.getcwd(), "logs", f"{args.name}_{split}_{i_split}")

        checkpoint = ModelCheckpoint(
            monitor="val/loss",
            save_last=True,
            mode="min",
            save_top_k=1,
            filename="best",
            dirpath=save_path,
        )
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            gpus=[args.device],
            callbacks=[checkpoint],
            default_root_dir=save_path,
            logger=wandb_logger,
        )

        # train
        print("Training...")
        trainer.fit(model,)

        # predict val
        print("Predicting val...")
        # _, cdr3_list, epi_list, _ = model.predict_valset()
        scores_samples = model.best_val_samples

        # save results
        save_path = os.path.join("results", args.name, split)
        os.makedirs(save_path, exist_ok=True)

        metric_df = pd.DataFrame(
            {
                "coef": scores_samples[0],
                "mae": scores_samples[1],
                "mape": scores_samples[2],
                "auc": scores_samples[3],
                "auprc": scores_samples[4],
            }
        )

        metric_df.to_csv(os.path.join(save_path, f"metrics_{i_split}.csv"))

        wandb.finish()
