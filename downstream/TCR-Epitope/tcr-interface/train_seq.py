import json
import math
import os
import sys

sys.path.append(".")
import argparse
import shutil

import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm

pl.seed_everything(0)
from collections import OrderedDict

import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from teim_utils import *
from torch import nn
from torch.utils.data import DataLoader

import mint
from mint.model.esm2 import ESM2


class SeqLevelSystem(pl.LightningModule):
    def __init__(self, model, args, train_set, val_set):
        super().__init__()

        self.model = model
        self.args = args
        self.train_set = train_set
        self.val_set = val_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.args.bs, shuffle=True, collate_fn=PPICollateFn()
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.args.bs, shuffle=False, collate_fn=PPICollateFn()
        )

    def forward(self, chains, chain_ids, epi_aa):
        return self.model(chains, chain_ids, epi_aa)

    def minimum_step(self, batch, device=None):

        if device is None:
            chains, chain_ids, target, epi_aa = batch
        else:
            chains, chain_ids, target, epi_aa = batch
            chains = chains.to(device)
            chain_ids = chain_ids.to(device)
            target = target.to(device)
            epi_aa = epi_aa.to(batch)

        pred = self.forward(chains, chain_ids, epi_aa)
        seq_pred = pred["seqlevel_out"]
        loss = self.get_loss(seq_pred, target)
        return loss, target, seq_pred

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss, labels, pred = self.minimum_step(batch)
        self.log("train/loss", loss)
        return {"loss": loss, "labels": labels, "pred": pred}

    def training_epoch_end(self, training_step_outputs):

        ## training metric
        # loss, auc, aupr, auc_mean, aupr_mean = self.evaluate_model(self.train_dataloader())

        # print('Train set: AUC={:.4}, AUPR={:.4}, AUC_AVG={:.4}, AUPR_AVG={:.4}'.format(auc, aupr, auc_mean, aupr_mean))
        # self.log('lr', self.model.optimizers().state_dict()['param_groups'][0]['lr'])
        # self.log_dict({
        #     'train/auc':auc,
        #     'train/aupr':aupr,
        #     'train/auc_avg':auc_mean,
        #     'train/aupr_avg':aupr_mean,
        # }, prog_bar=False)

        ## validating metric
        loss, auc, aupr, auc_mean, aupr_mean = self.evaluate_model(self.val_dataloader())

        print(loss, auc, aupr, auc_mean, aupr_mean)

        print(
            "Valid",
            " set: AUC={:.4}, AUPR={:.4}, AUC_AVG={:.4}, AUPR_AVG={:.4}".format(
                auc, aupr, auc_mean, aupr_mean
            ),
        )
        self.log_dict(
            {
                "valid/loss": loss,
                "valid/auc": auc,
                "valid/aupr": aupr,
                "valid/auc_avg": auc_mean,
                "valid/aupr_avg": aupr_mean,
            },
            prog_bar=False,
        )

    def evaluate_model(
        self, data_loader=None,
    ):
        self.model.eval()
        loss = 0
        y_true, y_pred = [], []
        epi_ids = []

        for i, batch in enumerate(tqdm(data_loader)):
            loss_this, y, y_hat = self.minimum_step(batch, self.device)
            loss += loss_this.item()
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(y_hat.detach().cpu().numpy().tolist())
            if "epi_id" in batch:
                epi_ids.extend(batch["epi_id"].cpu().numpy().tolist())
        loss /= i + 1
        auc, aupr = self.get_scores(y_true, y_pred)
        ## per epi auc
        if len(epi_ids) > 0:
            ids_uni = np.unique(epi_ids, axis=0)

            auc_sum = 0
            aupr_sum = 0
            cnt = 0
            for i, id_ in enumerate(ids_uni):
                index = np.array(epi_ids == id_)
                y_true_epi = np.array(y_true)[index]
                y_pred_epi = np.array(y_pred)[index]
                auc_epi, aupr_epi = self.get_scores(y_true_epi, y_pred_epi)
                if auc_epi is None:
                    continue
                auc_sum += auc_epi
                aupr_sum += aupr_epi
                cnt += 1
            auc_mean = auc_sum / cnt
            aupr_mean = aupr_sum / cnt
        else:
            auc_mean, aupr_mean = auc, aupr

        return loss, auc, aupr, auc_mean, aupr_mean

    def predict(self, data_loader=None):
        self.model.eval()
        cdr3_seqs, epi_seqs, y_true, y_pred = [], [], [], []
        epi_ids = []

        for i, batch in tqdm(enumerate(data_loader), desc="Predicting"):
            loss, y, y_hat = self.minimum_step(batch, self.device)
            # cdr3_seqs.extend(batch['cdr3_seqs'])
            # epi_seqs.extend(batch['epi_seqs'])
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(y_hat.detach().cpu().numpy().tolist())
            # if 'epi_id' in batch.keys():
            #     epi_ids.extend(batch['epi_id'].cpu().numpy().tolist())

        if len(epi_ids) > 0:
            return cdr3_seqs, epi_seqs, y_true, np.reshape(y_pred, -1), epi_ids
        else:
            return y_true, np.reshape(y_pred, -1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    def get_loss(self, pred, labels):
        loss = F.binary_cross_entropy(pred.view(-1), labels.float(), weight=None, reduction="mean")
        return loss

    def get_scores(self, y_true, y_pred):
        if len(np.unique(y_true)) == 1:
            return None, None
        else:
            return calc_auc_aupr(y_true, y_pred)


def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


class PPICollateFn:
    def __init__(self, truncation_seq_length=None):
        self.alphabet = mint.data.Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        len(batches)

        cdr3_list = [item["cdr3_seqs"] for item in batches]
        epi_list = [item["epi_seqs"] for item in batches]
        labels = [item["labels"] for item in batches]
        epi_aa_list = [item["epi"] for item in batches]

        cdr3_enc = self.convert(cdr3_list, 20 + 2)
        epi_enc = self.convert(epi_list, 12 + 2)

        chains = [cdr3_enc, epi_enc]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        labels = torch.from_numpy(np.stack(labels, 0))
        epi_aa = torch.from_numpy(np.stack(epi_aa_list, 0))

        return chains, chain_ids, labels, epi_aa

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

        if use_multimer:
            # remove 'model.' in keys
            new_checkpoint = OrderedDict(
                (key.replace("model.", ""), value)
                for key, value in checkpoint["state_dict"].items()
            )
            self.model.load_state_dict(new_checkpoint)
        else:
            new_checkpoint = upgrade_state_dict(checkpoint["model"])
            self.model.load_state_dict(new_checkpoint)
        total_layers = 33
        for name, param in self.model.named_parameters():
            if "embed_tokens.weight" in name or "_norm_after" in name or "lm_head" in name:
                param.requires_grad = False
            else:
                layer_num = name.split(".")[1]
                if int(layer_num) <= math.floor(total_layers * freeze_percent):
                    param.requires_grad = False

        ## feature extractor
        self.seq_cdr3 = nn.Sequential(
            nn.Conv1d(1280, dim_hidden, 1,), nn.BatchNorm1d(dim_hidden), nn.ReLU(),
        )
        self.seq_epi = nn.Sequential(
            nn.Conv1d(1280, dim_hidden, 1,), nn.BatchNorm1d(dim_hidden), nn.ReLU(),
        )

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

        if self.ae_model_cfg.path != "":
            ae_model = AutoEncoder(self.ae_model_cfg.dim_hid, self.ae_model_cfg.len_epi)
            self.ae_encoder = load_model_from_ckpt(self.ae_model_cfg.path, ae_model)
            for param in self.ae_encoder.parameters():
                param.requires_grad = False
            self.ae_linear = nn.Linear(self.ae_model_cfg.dim_hid, dim_hidden, bias=False)
        else:
            self.ae_encoder = None

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
    # parser.add_argument('--config', type=str, default='configs/seqlevel_cv_shuffle.yml')
    parser.add_argument("--config", type=str, default="./TEIM/train_teim/configs/seqlevel_all.yml")
    parser.add_argument("--freeze_percent", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cuda:7")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bs", type=int, default=48)

    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)

    config.data.path = "./TEIM/data/binding_data"
    config.model.ae_model.path = "./TEIM/ckpt/epi_ae.ckpt"
    # config.data.negative = 'original'

    cfg = argparse.Namespace()
    with open(
        f"/data/cb/scratch/varun/esm-multimer/esm-multimer/models/esm2_t33_650M_UR50D.json"
    ) as f:
        cfg.__dict__.update(json.load(f))
    esm_model = FlabWrapper(
        cfg,
        args.checkpoint_path,
        config.model.ae_model,
        args.freeze_percent,
        True,
        256,
        0.2,
        args.device,
    )

    datasets = load_data(config.data)
    train_set, val_set = datasets["train"], datasets["val"]

    for i_split, (train_set_this, val_set_this) in enumerate(zip(train_set, val_set)):
        print("Split {}".format(i_split), "Train:", len(train_set_this), "Val:", len(val_set_this))
        # load model and trainer
        print("Loading model and trainer...")
        model = SeqLevelSystem(esm_model, args, train_set_this, val_set_this)
        checkpoint = ModelCheckpoint(
            monitor="valid/auc_avg",
            save_last=True,
            mode="max",
            save_top_k=1,
            dirpath=os.path.join(os.getcwd(), "logs", "esm-m-tcr_seq"),
        )
        earlystop = EarlyStopping(monitor="valid/auc_avg", patience=15, mode="max")
        trainer = pl.Trainer(
            max_epochs=config.training.epochs,
            gpus=[7],
            callbacks=[checkpoint, earlystop],
            default_root_dir=os.path.join(os.getcwd(), "logs", "esm-m-tcr_seq"),
        )

        print(
            "Num of trainable parameters:",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        # train
        print("Training...")
        trainer.fit(model,)
        shutil.copy2(config_path, os.path.join(trainer.log_dir, os.path.basename(config_path)))

        # predict val
        print("Predicting val...")
        results = model.predict(model.val_dataloader())

        print(results)
