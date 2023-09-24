from .parts import *


class CountingModel(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(CountingModel, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels  # 32

        self.inc = SingleBlock(in_channels, mid_channels)
        self.res_block1 = ContracBlock(mid_channels, 2*mid_channels)
        self.res_block2 = ContracBlock(2*mid_channels, 4*mid_channels)
        self.res_block3 = ContracBlock(4*mid_channels, 8*mid_channels)
        self.res_block4 = ContracBlock(8*mid_channels, 8*mid_channels)
        
        self.res_block_exp1 = ExpBlock(8*mid_channels, 8*mid_channels)
        self.res_block_exp2 = ExpBlock(8*mid_channels, 4*mid_channels)
        self.res_block_exp3 = ExpBlock(4*mid_channels, 2*mid_channels)
        self.res_block_exp4 = ExpBlock(2*mid_channels, mid_channels)

        self.outc1 = OutBlock(mid_channels)
        self.outc2 = OutBlock(mid_channels)
        self.outc3 = OutBlock(mid_channels)
        self.outc4 = OutBlock(mid_channels)

        self.NonLocalBlock = NonLocalBlock(256)# 原文没有

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.res_block1(x1)
        x3 = self.res_block2(x2)
        x4 = self.res_block3(x3)
        x5 = self.res_block4(x4)
        y1 = self.res_block_exp1(x5, x4)
        y2 = self.res_block_exp2(y1, x3)
        y3 = self.res_block_exp3(y2, x2)
        y4 = self.res_block_exp4(y3, x1)
        out1 = self.outc1(y4)
        out2 = self.outc2(y4)
        out3 = self.outc3(y4)
        out4 = self.outc4(y4)
        return [out1, out2, out3, out4]

