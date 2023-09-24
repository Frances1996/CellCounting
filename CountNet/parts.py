import torch
import torch.nn as nn


class SingleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.singleblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.singleblock(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.downblock(x)


class DoubleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.doubleblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
            )
    def forward(self, x):
        return self.doubleblock(x)


class ContracBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downblock = DownBlock(in_channels, out_channels)
        self.doubleblock = DoubleBlock(out_channels, out_channels)
        
    def forward(self, x):
        x1 = self.downblock(x)
        return x1 + self.doubleblock(x1)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upblock = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.upblock(x)

class ExpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upblock = UpBlock(in_channels, out_channels)
        self.singleblock = SingleBlock(out_channels*2, out_channels)
        self.doubleblock = DoubleBlock(out_channels, out_channels)

    def forward(self, x1, x2):
        x = self.upblock(x1)
        x = torch.cat((x, x2), 1)
        x = self.singleblock(x)
        return x + self.doubleblock(x)

class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.outblock = SingleBlock(in_channels, in_channels)
        self.regblock = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.outblock(x)
        return self.regblock(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out