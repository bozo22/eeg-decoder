import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import Tensor
from einops.layers.torch import Rearrange

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.tsconv(x)
        x = self.projection(x)
        return x


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x


class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, **kwargs):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout_p):
        super().__init__()

        self.Attention1 = nn.MultiheadAttention(emb_dim, num_heads, dropout = dropout_p)
        self.Attention2 = nn.MultiheadAttention(emb_dim, num_heads, dropout = dropout_p)
        self.linear_expansion_dim = 4 * emb_dim

        # Add positional embeddings to EEG features
        self.pos_embedding = nn.Parameter(torch.randn(1, emb_dim))
        
        self.layer_norm11 = nn.RMSNorm(emb_dim)
        self.layer_norm12 = nn.RMSNorm(emb_dim)
        self.layer_norm21 = nn.RMSNorm(emb_dim)
        self.layer_norm22 = nn.RMSNorm(emb_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(emb_dim, self.linear_expansion_dim),
            nn.GELU(),
            nn.Linear(self.linear_expansion_dim, emb_dim),
            nn.Dropout(dropout_p)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(emb_dim, self.linear_expansion_dim),
            nn.GELU(),
            nn.Linear(self.linear_expansion_dim, emb_dim),
            nn.Dropout(dropout_p)
        )
        
        self.init_weights()

    def init_weights(self):
        # Initialize attention weights and possibly bias
        for attn in [self.Attention1, self.Attention2]:
            # Initialize the projection matrices with Xavier uniform
            nn.init.xavier_uniform_(attn.in_proj_weight)
            nn.init.xavier_uniform_(attn.out_proj.weight)
            
            # Initialize biases to zero
            if attn.in_proj_bias is not None:
                nn.init.constant_(attn.in_proj_bias, 0.)
            if attn.out_proj.bias is not None:
                nn.init.constant_(attn.out_proj.bias, 0.)
        # Initialize linear layers
        for mlp in [self.mlp1, self.mlp2]:
            for i, layer in enumerate(mlp):
                if isinstance(layer, nn.Linear):
                    # First layer (expansion): standard Xavier initialization
                    if i == 0:
                        nn.init.xavier_uniform_(layer.weight)
                    # Second layer (projection): scaled for residual connection
                    else:
                        nn.init.xavier_uniform_(layer.weight, gain=1/math.sqrt(2))
                    
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.)

    def forward(self, eeg_enc, image_enc):
        image_enc = self.layer_norm11(image_enc)
        eeg_enc = self.layer_norm12(eeg_enc)

        # Q from img features, K & V from EEG features
        image_enc = self.Attention1(image_enc, eeg_enc, eeg_enc)[0] + image_enc
        # Q from EEG features, K & V from img features
        eeg_enc = self.Attention2(eeg_enc, image_enc, image_enc)[0] + eeg_enc

        image_enc = self.layer_norm21(image_enc)
        eeg_enc = self.layer_norm12(eeg_enc)

        image_enc = self.mlp1(image_enc) + image_enc
        eeg_enc = self.mlp2(eeg_enc) + eeg_enc

        return eeg_enc, image_enc
    
class CrossAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout_p, n_blocks, use_attention=True):
        super().__init__()
        
        self.use_attention = use_attention
        if self.use_attention:
            attention_blocks = []
            for i in range(n_blocks):
                attention_blocks.append(CrossAttentionBlock(emb_dim, num_heads, dropout_p))
            self.attention_blocks = nn.Sequential(*attention_blocks)
        else:
            self.attention_blocks = nn.Identity()

    
    def forward(self, eeg_enc, image_enc):
        eeg_enc, image_enc = self.attention_blocks((eeg_enc, image_enc))
        return eeg_enc, image_enc
    
    def init_weights(self):
        if self.use_attention:
            for block in self.attention_blocks:
                block.init_weights()
