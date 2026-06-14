import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


class DepthwiseConv2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, bias=True, padding=1, groups=dim, stride=1
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DepthwiseConv2d(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, sr_ratio=1
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        head_dim = C // self.num_heads

        q = self.q(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)

        kv = kv.reshape(B, -1, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Strided conv with kernel > stride => overlapping patches, preserves
    local continuity (unlike ViT's non-overlapping patchify)."""

    def __init__(self, patch_size, stride, in_chans, embed_dim, norm_eps=1e-6):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=norm_eps)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """Hierarchical Mix Transformer encoder. Returns 4 multi-scale feature
    maps (one per stage), each at half the resolution of the previous."""

    def __init__(
        self,
        in_chans=3,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        norm_eps=1e-6,
    ):
        super().__init__()
        self.depths = depths

        # --- patch embeddings (one per stage) ---
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0], norm_eps)
        self.patch_embed2 = OverlapPatchEmbed(
            3, 2, embed_dims[0], embed_dims[1], norm_eps
        )
        self.patch_embed3 = OverlapPatchEmbed(
            3, 2, embed_dims[1], embed_dims[2], norm_eps
        )
        self.patch_embed4 = OverlapPatchEmbed(
            3, 2, embed_dims[2], embed_dims[3], norm_eps
        )

        # --- stochastic depth decay schedule across all blocks ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.block1 = nn.ModuleList(
            [
                Block(
                    embed_dims[0],
                    num_heads[0],
                    mlp_ratios[0],
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    dpr[cur + i],
                    sr_ratios[0],
                    norm_eps,
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = nn.LayerNorm(embed_dims[0], eps=norm_eps)
        cur += depths[0]

        self.block2 = nn.ModuleList(
            [
                Block(
                    embed_dims[1],
                    num_heads[1],
                    mlp_ratios[1],
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    dpr[cur + i],
                    sr_ratios[1],
                    norm_eps,
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = nn.LayerNorm(embed_dims[1], eps=norm_eps)
        cur += depths[1]

        self.block3 = nn.ModuleList(
            [
                Block(
                    embed_dims[2],
                    num_heads[2],
                    mlp_ratios[2],
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    dpr[cur + i],
                    sr_ratios[2],
                    norm_eps,
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = nn.LayerNorm(embed_dims[2], eps=norm_eps)
        cur += depths[2]

        self.block4 = nn.ModuleList(
            [
                Block(
                    embed_dims[3],
                    num_heads[3],
                    mlp_ratios[3],
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    dpr[cur + i],
                    sr_ratios[3],
                    norm_eps,
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = nn.LayerNorm(embed_dims[3], eps=norm_eps)

        self.apply(_init_weights)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


def mit_b1(drop_path_rate=0.1, **kwargs):
    """MiT-B1 configuration (used by SegFormer-B1)."""
    return MixVisionTransformer(
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=drop_path_rate,
        norm_eps=1e-6,
        **kwargs,
    )

# Decoder head

class MLP(nn.Module):
    """Per-stage linear projection: flattens a (B,C,H,W) feature map to
    (B,N,C), projects to embed_dim with a single Linear layer."""

    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    """1x1 conv + BatchNorm + ReLU, used to fuse the concatenated multi-scale
    features in the decode head."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SegFormerHead(nn.Module):
    """All-MLP decoder. Projects each encoder stage's features to a common
    embedding dim, upsamples all to the highest-resolution stage's size,
    concatenates, fuses with a 1x1 conv, then predicts per-pixel class
    logits at that (1/4 input) resolution."""

    def __init__(
        self,
        in_channels=(64, 128, 320, 512),
        embed_dim=256,
        num_classes=19,
        dropout_ratio=0.1,
    ):
        super().__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels

        self.linear_c4 = MLP(c4_in, embed_dim)
        self.linear_c3 = MLP(c3_in, embed_dim)
        self.linear_c2 = MLP(c2_in, embed_dim)
        self.linear_c1 = MLP(c1_in, embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        self.apply(_init_weights)

    def forward(self, features):
        c1, c2, c3, c4 = features
        target_size = c1.shape[2:]  # highest-resolution stage (stage 1)

        def project_and_resize(feat, linear):
            n, c, h, w = feat.shape
            x = linear(feat).permute(0, 2, 1).reshape(n, -1, h, w)
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            return x

        _c4 = project_and_resize(c4, self.linear_c4)
        _c3 = project_and_resize(c3, self.linear_c3)
        _c2 = project_and_resize(c2, self.linear_c2)
        _c1 = project_and_resize(
            c1, self.linear_c1
        )  # already at target_size, interpolate is a no-op

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(x)
        x = self.linear_pred(x)
        return x


class SegFormer(nn.Module):
    def __init__(self, num_classes=19, decoder_embed_dim=256, drop_path_rate=0.1):
        super().__init__()
        self.encoder = mit_b1(drop_path_rate=drop_path_rate)
        self.decode_head = SegFormerHead(
            in_channels=(64, 128, 320, 512),
            embed_dim=decoder_embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        out = self.decode_head(features)
        out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
        return out
