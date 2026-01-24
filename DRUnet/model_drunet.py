import torch
import torch.nn as nn

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(DilatedResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                      padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                      padding=dilation_rate, dilation=dilation_rate, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class DRUNet(nn.Module):
    """
    High-Capacity DRUNet.
    Increased filter depth to 1024 for maximum feature extraction potential.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(DRUNet, self).__init__()
        
        # Encoder - Increased Depth
        self.enc1 = DilatedResidualBlock(in_channels, 64, dilation_rate=1)
        self.enc2 = DilatedResidualBlock(64, 128, dilation_rate=2)
        self.enc3 = DilatedResidualBlock(128, 256, dilation_rate=2)
        self.enc4 = DilatedResidualBlock(256, 512, dilation_rate=2)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck - Full Potential (1024 filters)
        self.bottleneck = DilatedResidualBlock(512, 1024, dilation_rate=4)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DilatedResidualBlock(1024, 512, dilation_rate=2)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DilatedResidualBlock(512, 256, dilation_rate=2)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DilatedResidualBlock(256, 128, dilation_rate=2)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DilatedResidualBlock(128, 64, dilation_rate=1)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        
        b = self.bottleneck(self.pool(s4))
        
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([s4, d4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([s3, d3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([s2, d2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([s1, d1], dim=1))
        
        return self.final_conv(d1)