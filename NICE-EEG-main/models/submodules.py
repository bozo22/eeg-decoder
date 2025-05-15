"""
Different EEG encoders for comparison

SA GA

shallownet, deepnet, eegnet, conformer, tsconv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv

from einops import rearrange
from einops.layers.torch import Rearrange


class channel_attention(nn.Module):
    def __init__(self, sequence_num=250, inter=30):
        super(channel_attention, self).__init__()
        self.sequence_num = sequence_num
        self.inter = inter
        self.extract_sequence = int(
            self.sequence_num / self.inter
        )  # You could choose to do that for less computation

        self.query = nn.Sequential(
            nn.Linear(63, 63),
            nn.LayerNorm(63),  # also may introduce improvement to a certain extent
            nn.Dropout(0.3),
        )
        self.key = nn.Sequential(
            nn.Linear(63, 63),
            # nn.LeakyReLU(),
            nn.LayerNorm(63),
            nn.Dropout(0.3),
        )
        # self.value = self.key
        self.projection = nn.Sequential(
            nn.Linear(63, 63),
            # nn.LeakyReLU(),
            nn.LayerNorm(63),
            nn.Dropout(0.3),
        )
        self.drop_out = nn.Dropout(0)
        self.pooling = nn.AvgPool2d(kernel_size=(1, self.inter), stride=(1, self.inter))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        temp = rearrange(x, "b o c s->b o s c")
        temp_query = rearrange(self.query(temp), "b o s c -> b o c s")
        temp_key = rearrange(self.key(temp), "b o s c -> b o c s")

        channel_query = temp_query
        channel_key = temp_key

        scaling = self.extract_sequence ** (1 / 2)

        channel_atten = (
            torch.einsum("b o c s, b o m s -> b o c m", channel_query, channel_key)
            / scaling
        )

        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.drop_out(channel_atten_score)

        out = torch.einsum("b o c s, b o c m -> b o c s", x, channel_atten_score)
        """
        projections after or before multiplying with attention score are almost the same.
        """
        out = rearrange(out, "b o c s -> b o s c")
        out = self.projection(out)
        out = rearrange(out, "b o s c -> b o c s")
        return out


class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(
            in_channels=in_channels, out_channels=out_channels, heads=1
        )
        self.num_channels = 63
        # Create a list of tuples representing all possible edges between channels
        edge_index_list = torch.Tensor(
            [
                (i, j)
                for i in range(self.num_channels)
                for j in range(self.num_channels)
                if i != j
            ]
        )
        # Convert the list of tuples to a tensor
        self.register_buffer("edge_index", torch.tensor(edge_index_list, dtype=torch.long).t().contiguous())

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()
        # Reshape x to (batch_size*num_channels, num_features) to pass through GATConv
        x = x.view(batch_size * num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
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


# ===== Patch Encoder =====

class MultiScaleTemporalConvBlock(nn.Module):
    """
    Inception-style multi-scale temporal convolution for EEG.

    Options
    -------
    • dilation per branch
    • optional AvgPool for temporal down-sampling
    • optional Squeeze-and-Excitation (channel attention)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_sizes=(3, 11, 25, 25),
        dilation_rates=(1, 1, 1, 2),
        pool_cfg=dict(kernel_size=(1, 51), stride=(1, 5)),  # set stride>1 to down-sample
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilation_rates), "kernel_sizes and dilation_rates must match"
        n_branches = len(kernel_sizes)
        assert out_ch % n_branches == 0, "out_ch must be divisible by #branches"
        branch_ch = out_ch // n_branches

        # parallel temporal conv branches
        self.branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilation_rates):
            pad = ((k - 1) * d) // 2  # ensures same temporal length
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch,
                        branch_ch,
                        kernel_size=(1, k),
                        stride=(1, 1),
                        padding=(0, pad),
                        dilation=(1, d),
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_ch),
                    nn.ELU(inplace=True),
                )
            )

        # feature-level channel-attention (SE)
        self.se = SqueezeExcite(out_ch, reduction=8)

        # 1×1 conv to mix branch features (acts like a pointwise projection)
        self.mix = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True),
        )

        # optional temporal pooling / down-sampling
        self.pool = nn.AvgPool2d(**pool_cfg) if pool_cfg["stride"] != (1, 1) else nn.Identity()

        # residual path (identity or 1×1 projection)
        if in_ch == out_ch:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, in_ch, C=63, T]
        branch_outs = [b(x) for b in self.branches]     # list of [B, branch_ch, 63, T]
        x_cat = torch.cat(branch_outs, dim=1)           # [B, out_ch, 63, T]
        x_cat = self.mix(x_cat)                         # pointwise fusion
        x_cat = self.se(x_cat)                          # feature-level channel attention
        x_cat = self.pool(x_cat)                        # optional down-sampling
        x_cat = self.dropout(x_cat)

        # residual add (handles input and output channel mismatch automatically)
        # return nn.ELU(inplace=True)(x_cat + self.residual(self.pool(x))) # interesting as next step
        return x_cat



class SqueezeExcite(nn.Module):
    """Classic SE block (feature-level channel attention) with reduction ratio."""
    def __init__(self, chan, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(chan, chan // reduction, 1, bias=False),
            nn.ELU(inplace=True),
            nn.Conv2d(chan // reduction, chan, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(self.pool(x))
        return x * scale


class Debugger(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x: Tensor) -> Tensor:
        print(f"{self.name} x.shape: {x.shape}")
        return x

# ===== Other useful modules =====

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()
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
        self.deepnet = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), (1, 1)),
            nn.Conv2d(25, 25, (63, 1), (1, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(25, 50, (1, 10), (1, 1)),
            nn.BatchNorm2d(50),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(50, 100, (1, 10), (1, 1)),
            nn.BatchNorm2d(100),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(100, 200, (1, 10), (1, 1)),
            nn.BatchNorm2d(200),
            nn.ELU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
        )

        self.eegnet = nn.Sequential(
            nn.Conv2d(1, 8, (1, 64), (1, 1)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (63, 1), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 16, (1, 16), (1, 1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # nn.AvgPool2d((1, 2), (1, 2)),
            nn.Dropout2d(0.5),
        )

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # 5 is better than 1
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.tsconv(x)
        return x
    

class ClassificationHead(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 40),
        )

    def forward(self, x):
        cls_out = self.fc(x)
        return x, cls_out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum(
            "bhqd, bhkd -> bhqk", queries, keys
        )  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum("bhal, bhlv -> bhav ", att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        num_heads=10,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size, expansion=forward_expansion, drop_p=forward_drop_p
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])