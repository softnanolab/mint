from omegaconf import DictConfig
import lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from mint.model.modules import MINTContactHead
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

        # create MLP head here
        self.contact_head = MINTContactHead(cfg.mint.esm2.encoder_embed_dim)

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

        tokens, chain_ids, contact_masks = batch
        # Build MLM mask (exclude CLS/EOS/PAD from potential masking)
        mask = (
            (~tokens.eq(self.model.cls_idx))
            & (~tokens.eq(self.model.eos_idx))
            & (~tokens.eq(self.model.padding_idx))
        )
        mask = (torch.rand(tokens.shape, device=tokens.device) < 0.15) & mask

        # Prepare masked input
        rand = torch.rand(tokens.shape, device=tokens.device)
        randaa = torch.randint(4, 24, tokens.shape, device=tokens.device)

        inp = tokens
        inp = torch.where((rand < 0.8) & mask, self.model.mask_idx, inp)
        inp = torch.where((rand > 0.9) & mask, randaa, inp)

        # Single forward pass to get logits and representations
        model_out = self.model(inp, chain_ids)

        # Get final-layer embeddings (B x L x E) without hardcoding the layer index
        final_reps = model_out["representations"][self.model.num_layers]

        # Mask out embeddings that relate to special tokens: CLS, EOS, PAD
        special_mask = (
            (tokens == self.model.cls_idx)
            | (tokens == self.model.eos_idx)
            | (tokens == self.model.padding_idx)
        )  # (B x L)
        # Optionally also exclude masked MLM tokens by uncommenting the next line
        # special_mask = special_mask | (tokens == self.model.mask_idx)

        # Zero out the special-token positions while keeping the (B x L x E) shape
        final_reps = final_reps.masked_fill(special_mask.unsqueeze(-1), 0.0)

        # Compact valid (non-special) tokens per sequence.
        valid_mask = ~special_mask  # (B x L)
        B, L, E = final_reps.shape
        if B == 1:
            # No padding: (1 x N_valid x E)
            filtered_reps = final_reps[0, valid_mask[0]].unsqueeze(0)
        else:
            # Padded compaction: (B x N_valid_max x E)
            lengths = valid_mask.sum(dim=1)  # (B)
            max_len = lengths.max().clamp_min(1)
            sort_idx = torch.argsort(valid_mask.int(), dim=1, descending=True)  # (B x L)
            gathered = torch.gather(
                final_reps, 1, sort_idx.unsqueeze(-1).expand(-1, -1, E)
            )  # (B x L x E)
            filtered_reps = gathered[:, : max_len.item(), :]

        # Build pairwise tensor: (B x N_valid x N_valid x E) where final[:, i, j, :] = filtered[:, i, :]
        N_valid = filtered_reps.size(1)
        pair_reps = filtered_reps.unsqueeze(2).repeat(1, 1, N_valid, 1)

        # Feed pair_reps into the contact head
        predicted_contact_masks = self.contact_head.forward(pair_reps)

        # Compute the Binary Cross Entropy Loss for the predicted contacts
        # Build mask of valid entries (not ignored)
        valid_entries = contact_masks != -1

        # Compute BCE loss per element (no reduction)
        loss_per_elem = F.binary_cross_entropy_with_logits(
            predicted_contact_masks, contact_masks, reduction="none"
        )

        # Apply mask and normalize
        loss_contact_head = (loss_per_elem * valid_entries).sum() / (valid_entries.sum())

        # Compute MLM loss
        logits = model_out["logits"]
        loss_mlm = F.cross_entropy(logits.transpose(1, 2), tokens, reduction="none")
        loss_mlm = (loss_mlm * mask).sum() / mask.sum()

        # Compute the averaged loss for both the MLM and Supervised Contact Mask Prediction Task
        loss = loss_mlm + loss_contact_head

        # Return loss and pairwise embeddings (B x L x L x E)
        return loss, filtered_reps

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
