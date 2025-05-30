import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1,filter=64):
        super().__init__()
        
        self.filter = filter
        # Encoder
        self.down1 = DoubleConv(in_channels, self.filter * 2**0)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(self.filter * 2**0 , self.filter * 2**1)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(self.filter * 2**1, self.filter * 2**2)
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(self.filter * 2**2, self.filter * 2**3)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(self.filter * 2**3, self.filter * 2**4)

        # Decoder
        self.up4 = nn.ConvTranspose2d(self.filter * 2**4, self.filter * 2**3, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(self.filter * 2**4, self.filter * 2**3)

        self.up3 = nn.ConvTranspose2d(self.filter * 2**3, self.filter * 2**2, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(self.filter * 2**3, self.filter * 2**2)

        self.up2 = nn.ConvTranspose2d(self.filter * 2**2, self.filter * 2**1, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(self.filter * 2**2, self.filter * 2**1)

        self.up1 = nn.ConvTranspose2d(self.filter * 2**1, self.filter * 2**0, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(self.filter * 2**1, self.filter * 2**0)

        # Final output
        self.out_conv = nn.Conv2d(self.filter * 2**0, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.down4(self.pool3(x3))

        # Bottleneck
        x5 = self.bottleneck(self.pool4(x4))

        # Decoder
        x = self.up4(x5)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return torch.sigmoid(self.out_conv(x))
