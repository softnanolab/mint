from pathlib import Path
from omegaconf import OmegaConf

from mint.model.mint import MINT

BASE_PATH = Path(__file__).parent.parent


def test_mint():
    cfg = OmegaConf.load(BASE_PATH / "src/mint/configs/main.yaml")
    # override for faster testing
    cfg.mint.esm2.encoder_layers = 2
    cfg.mint.esm2.encoder_embed_dim = 64
    cfg.mint.esm2.encoder_attention_heads = 4
    # load the model
    model = MINT(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params} parameters")
    assert num_params == 135779, f"Model should have parameters, got {num_params}"
