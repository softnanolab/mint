from omegaconf import DictConfig

import lightning as pl
import torch

from mint.model.esm import ESM2


class MINT(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # TODO: check if this is needed here
        self.save_hyperparameters(cfg)
        self.model = ESM2(
            num_layers=cfg.mint.esm2.encoder_layers,
            embed_dim=cfg.mint.esm2.encoder_embed_dim,
            attention_heads=cfg.mint.esm2.encoder_attention_heads,
            token_dropout=cfg.mint.esm2.token_dropout,
            use_multimer=cfg.mint.esm2.use_multimer,
        )

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        # Manual checkpointing
        # if self.iter_step % 15000 == 0:
        # if self.trainer.is_global_zero:
        # torch.save(self.model.state_dict(), f'./workdir/3B_nofreeze/checkpoint_iter_{self.iter_step}.pt')
        self.log("train/loss", loss)
        # TODO: not sure if I should be doing this here
        # self.log("train/perplexity", torch.exp(loss))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, out = self.forward(batch)
        self.log("val/loss", loss)
        self.log("val/perplexity", torch.exp(loss))
        return loss

    def forward(self, batch):
        # 15% of tokens randomly sampled from the sequence. For those 15% of tokens, we change the input token to a special “masking”
        # token with 80% probability, a randomly-chosen alternate amino acid token with 10% probability, and the original input token
        # (i.e. no change) with 10% probability. We take the loss to be the whole batch average cross entropy loss between the model’s
        # predictions and the true token for these 15% of amino acid tokens.

        tokens, chain_ids = batch
        mask = (
            (~tokens.eq(self.model.cls_idx))
            & (~tokens.eq(self.model.eos_idx))
            & (~tokens.eq(self.model.padding_idx))
        )
        mask = (torch.rand(tokens.shape, device=tokens.device) < 0.15) & mask

        rand = torch.rand(tokens.shape, device=tokens.device)
        randaa = torch.randint(4, 24, tokens.shape, device=tokens.device)

        inp = tokens
        inp = torch.where((rand < 0.8) & mask, self.model.mask_idx, inp)
        inp = torch.where((rand > 0.9) & mask, randaa, inp)

        out = self.model(inp, chain_ids)["logits"]
        loss = torch.nn.functional.cross_entropy(out.transpose(1, 2), tokens, reduction="none")
        loss = (loss * mask).sum() / mask.sum()

        # self.log("tokens", mask.sum())
        # self.log("loss", loss)
        # self.log("perplexity", torch.exp(loss))
        return loss, out

    def configure_optimizers(self):
        # For model training optimization, we used Adam with 𝛽𝛽1 = 0.9, 𝛽𝛽2 = 0.98, 𝜖𝜖 = 10−8 and 𝐿𝐿2 weight decay of
        # 0.01 for all models except the 15 billion parameter model, where we used a weight decay of 0.1. The learning rate is
        # warmed up over the first 2,000 steps to a peak value of 4e-4 (1.6e-4 for the 15B parameter model), and then linearly
        # decayed to one tenth of its peak value over the 90% of training duration
        if self.cfg.training_args.freeze_self_attn:
            self.model.requires_grad_(False)
            for name, p in self.model.named_parameters():
                if "multimer_attn" in name:
                    p.requires_grad = True

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.training_args.lr,
            betas=self.cfg.training_args.adam_betas,
            eps=self.cfg.training_args.adam_eps,
            weight_decay=self.cfg.training_args.weight_decay,
        )

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-12,
            end_factor=1.0,
            total_iters=self.cfg.training_args.warmup_updates,
        )
        decay = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=self.cfg.training_args.end_learning_rate / self.cfg.training_args.lr,
            total_iters=int(0.9 * int(self.cfg.training_args.total_num_update)),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, decay],
            milestones=[self.cfg.training_args.warmup_updates],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
