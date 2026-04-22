import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TinyUNet(nn.Module):
    """Simplified U-Net for multi-label segmentation"""

    def __init__(self, n_channels=3, n_classes=2, bilinear=True):
        """
        Args:
            n_channels: Number of input channels (3 for RGB)
            n_classes: Number of output channels (2 for roof and solar)
            bilinear: Use bilinear upsampling
        """
        super().__init__()

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output layer
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ResNetUNet(nn.Module):
    """U-Net with pretrained ResNet34 encoder for better feature extraction"""

    def __init__(self, n_classes=2, pretrained=True):
        super().__init__()
        from torchvision.models import resnet34, ResNet34_Weights

        encoder = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        # Encoder stages (reuse ResNet layers)
        self.enc0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)  # /2, 64ch
        self.pool0 = encoder.maxpool                                          # /4
        self.enc1 = encoder.layer1  # /4, 64ch
        self.enc2 = encoder.layer2  # /8, 128ch
        self.enc3 = encoder.layer3  # /16, 256ch
        self.enc4 = encoder.layer4  # /32, 512ch

        # Decoder (upsample + concat skip + conv)
        self.up4 = DoubleConv(512 + 256, 256)
        self.up3 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.up1 = DoubleConv(64 + 64, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)       # /2, 64ch
        e1 = self.enc1(self.pool0(e0))  # /4, 64ch
        e2 = self.enc2(e1)      # /8, 128ch
        e3 = self.enc3(e2)      # /16, 256ch
        e4 = self.enc4(e3)      # /32, 512ch

        # Decoder with skip connections
        d4 = self.up4(torch.cat([F.interpolate(e4, e3.shape[2:], mode='bilinear', align_corners=True), e3], dim=1))
        d3 = self.up3(torch.cat([F.interpolate(d4, e2.shape[2:], mode='bilinear', align_corners=True), e2], dim=1))
        d2 = self.up2(torch.cat([F.interpolate(d3, e1.shape[2:], mode='bilinear', align_corners=True), e1], dim=1))
        d1 = self.up1(torch.cat([F.interpolate(d2, e0.shape[2:], mode='bilinear', align_corners=True), e0], dim=1))

        # Back to original resolution
        out = F.interpolate(d1, x.shape[2:], mode='bilinear', align_corners=True)
        return self.outc(out)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    x = torch.randn(2, 3, 896, 896).to(device)

    # Test TinyUNet
    model = TinyUNet(n_channels=3, n_classes=2).to(device)
    y = model(x)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TinyUNet    - Output: {y.shape}, Params: {n_params:,}")

    # Test ResNetUNet
    model2 = ResNetUNet(n_classes=2, pretrained=True).to(device)
    y2 = model2(x)
    n_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"ResNetUNet  - Output: {y2.shape}, Params: {n_params2:,}")
