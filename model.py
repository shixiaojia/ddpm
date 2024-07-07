import math
import torch
import torch.nn as nn
from torch.nn import init


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0, "error d_model!"
        super(TimeEmbedding, self).__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        X = pos[:, None] * emb[None, :]
        emb = torch.zeros(T, d_model)
        emb[:, 0::2] = torch.sin(X)
        emb[:, 1::2] = torch.cos(X)

        self.time_embedding = nn.Sequential(nn.Embedding.from_pretrained(emb),
                                            nn.Linear(d_model, dim),
                                            Swish(),
                                            nn.Linear(dim, dim))
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.time_embedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super(DownSample, self).__init__()
        self.down = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.down.weight)
        init.zeros_(self.down.bias)

    def forward(self, x, temb):
        x = self.down(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch, kernel_size=2, stride=2, padding=0)
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.conv.weight)
        init.zeros_(self.conv.bias)

        init.xavier_uniform_(self.up.weight)
        init.zeros_(self.up.bias)

    def forward(self, x, temb):
        x = self.up(x)
        x = self.conv(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super(AttnBlock, self).__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        N, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(N, H * W, C)
        k = k.view(N, C, H * W)
        score = q @ k * (C**-0.5) # N, H*W, H*W
        score = score.softmax(dim=-1)
        v = v.permute(0, 2, 3, 1).view(N, H*W, C)
        h = score @ v
        h = h.view(N, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        return x + h


# DownBlock = ResBlock + AttnBlock
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, dropout, attn=False):
        super(ResBlock, self).__init__()
        self.block1 = nn.Sequential(nn.GroupNorm(32, in_ch),
                                    Swish(),
                                    nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1))

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(t_dim, out_ch)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        )

        if in_ch != out_ch:
            self.short_cut = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.short_cut = nn.Identity()

        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):
        h = self.block1(x)
        h += self.temb_proj(temb)[..., None, None]
        h = self.block2(h)
        h += self.short_cut(x)

        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_ratio, num_res_block, dropout):
        super(UNet, self).__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)

        self.head = nn.Conv2d(in_channels=3, out_channels=ch, kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList()
        chs = [ch]
        in_ch = ch
        for i, ratio in enumerate(ch_ratio):
            out_ch = ch * ratio
            for _ in range(num_res_block):
                self.down_blocks.append(ResBlock(in_ch=in_ch, out_ch=out_ch, t_dim=tdim,
                                                 dropout=dropout, attn=True))
                in_ch = out_ch
                chs.append(in_ch)

            if i != len(ch_ratio) - 1:
                self.down_blocks.append(DownSample(in_ch=in_ch))
                chs.append(in_ch)

        self.middle_blocks = nn.ModuleList([ResBlock(in_ch=in_ch, out_ch=in_ch, t_dim=tdim, dropout=dropout, attn=True),
                                            ResBlock(in_ch=in_ch, out_ch=in_ch, t_dim=tdim, dropout=dropout, attn=False)])

        self.up_blocks = nn.ModuleList()

        for i, ratio in reversed(list(enumerate(ch_ratio))):
            out_ch = ch * ratio
            for _ in range(num_res_block+1):
                self.up_blocks.append(ResBlock(in_ch=chs.pop()+in_ch, out_ch=out_ch, t_dim=tdim, dropout=dropout, attn=True))
                in_ch = out_ch

            if i != 0:
                self.up_blocks.append(UpSample(in_ch=in_ch))

        self.tail = nn.Sequential(nn.GroupNorm(32, in_ch),
                                  Swish(),
                                  nn.Conv2d(in_channels=in_ch, out_channels=3, kernel_size=3, stride=1, padding=1))

        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)

        init.xavier_uniform_(self.tail[-1].weight)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t):
        temb = self.time_embedding(t)
        h = self.head(x)
        # down
        hs = [h]
        for layer in self.down_blocks:
            h = layer(h, temb)
            hs.append(h)

        # middle
        for layer in self.middle_blocks:
            h = layer(h, temb)

        # up
        for layer in self.up_blocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        h = self.tail(h)
        return h
