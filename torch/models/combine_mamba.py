import torch
import torch.nn as nn

#mamba
from .mamba import MambaLayer

#monai
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class GSC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.proj(x)
        x2 = self.skip(x)
        x = self.final(x1 + x2)
        return x + x

class MambaEncoder2D(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384]):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Conv2d(in_chans, dims[0], kernel_size=7, stride=2, padding=3))
        for i in range(3):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.InstanceNorm2d(dims[i]),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
            )
        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(4):
            self.gscs.append(GSC(dims[i]))
            self.stages.append(nn.Sequential(*[MambaLayer(dim=dims[i]) for _ in range(depths[i])]))
            self.mlps.append(nn.Sequential(
                nn.InstanceNorm2d(dims[i]),
                MlpChannel(dims[i], dims[i] * 2),
            ))

    def forward(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            outs.append(self.mlps[i](x))
        return tuple(outs)

class Mamba2D(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, depths=[2, 2, 2, 2], feat_size=[48, 96, 192, 384], hidden_size=768):
        super().__init__()
        # feat_size=[96, 192, 384, 768]
        self.encoder = MambaEncoder2D(in_chans, depths, feat_size)
        # hidden_size=768*2
        self.encoder1 = UnetrBasicBlock(2, in_chans, feat_size[0], 3, 1, "instance", True)
        self.encoder2 = UnetrBasicBlock(2, feat_size[0], feat_size[1], 3, 1, "instance", True)
        self.encoder3 = UnetrBasicBlock(2, feat_size[1], feat_size[2], 3, 1, "instance", True)
        self.encoder4 = UnetrBasicBlock(2, feat_size[2], feat_size[3], 3, 1, "instance", True)
        self.encoder5 = UnetrBasicBlock(2, feat_size[3], hidden_size, 3, 1, "instance", True)

        self.decoder5 = UnetrUpBlock(2, hidden_size, feat_size[3], 3, 2, "instance", True)
        self.decoder4 = UnetrUpBlock(2, feat_size[3], feat_size[2], 3, 2, "instance", True)
        self.decoder3 = UnetrUpBlock(2, feat_size[2], feat_size[1], 3, 2, "instance", True)
        self.decoder2 = UnetrUpBlock(2, feat_size[1], feat_size[0], 3, 2, "instance", True)

        self.out = UnetOutBlock(2, feat_size[0], out_chans)

    def forward(self, x):
        vit_feats = self.encoder(x)
        # print(vit_feats[0].shape)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(vit_feats[0])
        enc3 = self.encoder3(vit_feats[1])
        enc4 = self.encoder4(vit_feats[2])
        enc5 = self.encoder5(vit_feats[3])

        dec5 = self.decoder5(enc5, enc4)
        dec4 = self.decoder4(dec5, enc3)
        dec3 = self.decoder3(dec4, enc2)
        dec2 = self.decoder2(dec3, enc1)

        return torch.sigmoid(self.out(dec2))