import torch.nn as nn

import torch as th
from torch import Tensor
from einops.layers.torch import Rearrange
from models.submodules import Debugger, MultiScaleTemporalConvBlock, ResidualAdd, EEG_GAT, SqueezeExcite, channel_attention, FlattenHead, NUM_ELECTRODES


# ===== EEG Encoder =====
class EEG_Denoiser(nn.Module):
    def __init__(self, dim=250, n_aggregations=4, mlp_ratio=4):
        super().__init__()
        self.denoiser = nn.Sequential(
            Rearrange("b a c s -> b c (a s)"),
            nn.Linear(dim * n_aggregations, dim * mlp_ratio),
            nn.ELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.denoiser(x).unsqueeze(1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, config="GA", patch_encoder="tsconv"):
        super().__init__(
            # nn.InstanceNorm2d(num_features=1), # Normalize each trial
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
            final_channels = 40
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
        elif type == "multiscale_1block":
            final_channels = 42
            self.patch_encoder = nn.Sequential(
                MultiScaleTemporalConvBlock(
                    in_ch=1,
                    out_ch=final_channels,
                    kernel_sizes=[3, 15, 25],
                    dilation_rates=[1, 1, 2],
                    pool_cfg=dict(
                        kernel_size=(1, 51),
                        stride=(1, 5)
                    ),
                    dropout_p=0.3,
                ),

                # Aggregation across electrodes
                nn.Conv2d(final_channels, final_channels, (63, 1), (1, 1)),
                nn.BatchNorm2d(final_channels),
                nn.ELU(),
                nn.Dropout2d(0.3),
            )
        elif type == "multiscale_2block":
            intermediate_channels = 33
            final_channels = 42
            self.patch_encoder = nn.Sequential(
                # Temporal block 1
                MultiScaleTemporalConvBlock(
                    in_ch=1,
                    out_ch=intermediate_channels,
                    kernel_sizes=(3, 11, 21),
                    dilation_rates=(1, 2, 3),
                    pool_cfg=dict(
                        kernel_size=(1, 30),
                        stride=(1, 2)
                    ),
                    dropout_p=0.25,
                ),

                # Spatial electrode-level SE & mixing across electrodes
                Rearrange("b c h w -> b h c w"),
                SqueezeExcite(NUM_ELECTRODES),
                nn.Conv2d(NUM_ELECTRODES, NUM_ELECTRODES, 1, bias=False),
                nn.BatchNorm2d(NUM_ELECTRODES),
                nn.ELU(inplace=True),
                Rearrange("b h c w -> b c h w"),

                # Temporal block 2
                MultiScaleTemporalConvBlock(
                    in_ch=intermediate_channels,
                    out_ch=final_channels,
                    kernel_sizes=(3, 9, 15),
                    dilation_rates=(1, 2, 3),
                    pool_cfg=dict(
                        kernel_size=(1, 7),
                        stride=(1, 3)
                    ),
                    dropout_p=0.25,
                ),

                # Spatial compacting
                nn.Conv2d(final_channels, final_channels, (63, 1), (1, 1)),
                nn.BatchNorm2d(final_channels),
                nn.ELU(),
            )
        
        self.projection = nn.Sequential(
            nn.Conv2d(final_channels, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
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
