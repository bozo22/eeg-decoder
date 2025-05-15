import torch.nn as nn

import torch as th
from torch import Tensor
from einops.layers.torch import Rearrange
from models.submodules import Debugger, MultiScaleTemporalConvBlock, ResidualAdd, EEG_GAT, channel_attention, FlattenHead


# ===== EEG Encoder =====
class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, config="GA", patch_encoder="tsconv", **kwargs):
        super().__init__(
            SpatialEncoder(config),
            PatchEmbedding(emb_size, patch_encoder),
            FlattenHead(),
        )


class SpatialEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.spatial = nn.Identity()
        if config == "SA":
            self.spatial = ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(250),
                    channel_attention(),
                    nn.Dropout(0.3),
                )
            )
        elif config == "GA":
            self.spatial = ResidualAdd(
                nn.Sequential(
                    EEG_GAT(),
                    nn.Dropout(0.3),
                )
            )
        elif config == "SAGA":
            self.spatial = ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(250),
                    channel_attention(),
                    nn.Dropout(0.3),
                    EEG_GAT(),
                    nn.Dropout(0.3),
                )
            )
        elif config == "GASA":
            self.spatial = ResidualAdd(
                nn.Sequential(
                    EEG_GAT(),
                    nn.Dropout(0.3),
                    nn.LayerNorm(250),
                    channel_attention(),
                    nn.Dropout(0.3),
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.spatial(x)

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, type="tsconv"):
        super().__init__()
        if type == "tsconv":
        # revised from shallownet
            self.patch_encoder = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (63, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )
        elif type == "multiscale":
            self.patch_encoder = nn.Sequential(
                MultiScaleTemporalConvBlock(in_ch=1, out_ch=40),
                nn.Conv2d(40, 40, (63, 1), (1, 1), groups=40),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.patch_encoder(x)
        x = self.projection(x)
        return x

# ===== Projectors =====


class Proj_eeg(nn.Sequential):
    def __init__(self, input_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(input_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(
        self, input_dim=768, proj_dim=768, drop_proj=0.3, use_image_projector=False
    ):
        self.use_image_projector = use_image_projector
        super().__init__(
            nn.Linear(input_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        if self.use_image_projector:
            return super().forward(x)
        else:
            return x
