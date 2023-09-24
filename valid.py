import torch
from torchvision.transforms import functional as F
import os
from utils import Adder

def valid(net, val_data):
    net.eval()
    mse_adder = Adder()
    criteria = torch.nn.MSELoss()
    with torch.no_grad():
        for i, data in enumerate(val_data):
            image, mask0, mask1, mask2, mask3 = data
            image = image.cuda()
            mask0 = mask0.cuda()
            mask1 = mask1.cuda()
            mask2 = mask2.cuda()
            mask3 = mask3.cuda()
            out = net(image)
            mse0 = criteria(out[0], mask0)
            mse1 = criteria(out[1], mask1)
            mse2 = criteria(out[2], mask2)
            mse3 = criteria(out[3], mask3)
            mse = mse0 + mse1 + mse2 + mse3
            mse_adder(mse)

    print('\n')
    net.train()
    return mse_adder.average()