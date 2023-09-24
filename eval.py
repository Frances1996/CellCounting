import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from data import test_dataloader
from utils import Adder


class PairFill(object):
    # 将图片resize或者使用黑色填充为固定大小,图像变换，没有扩增
    def __init__(self, test_size):
        self.img_size = test_size
    def __call__(self, img, label):
        h, w = img.shape[1], img.shape[2]
        transform = transforms.Resize(size=(self.img_size, self.img_size))
        if h >= self.img_size and w >= self.img_size:
            out = transform(img)
            out_label = transform(label)
        else:
            length = max(self.img_size, h, w)
            out = torch.zeros(img.shape[0], length, length)
            lefttop = [int(length/2-h/2), int(length/2-w/2)]
            out[:, lefttop[0]:lefttop[0]+h, lefttop[1]:lefttop[1]+w] = img
            out = transform(out)
            out_label = torch.zeros(label.shape[0], length, length)
            out_label[:, lefttop[0]:lefttop[0]+h, lefttop[1]:lefttop[1]+w] = label
            out_label = transform(out_label)
        return out, out_label

def eval(net, args):
    state_dict = torch.load(args.test_model)
    net.load_state_dict(state_dict['model'])
    dataloader = test_dataloader(args.path, args.kernel_size, args.alpha, args.n_classes, batch_size=1, num_workers=0)
    net.eval()
    criteria = torch.nn.MSELoss()
    with torch.no_grad():
        mse_adder = Adder()
        for i, data in enumerate(dataloader):
            img, label, image_name = data
            h, w = img.shape[2], img.shape[3]
            transform = transforms.Resize(size=(args.test_size, args.test_size))
            length = max(args.test_size, h, w)
            input_img = torch.zeros(img.shape[0], img.shape[1], length, length)
            lefttop = [int(length/2-h/2), int(length/2-w/2)]
            input_img[:, :, lefttop[0]:lefttop[0]+h, lefttop[1]:lefttop[1]+w] = img
            input_img = transform(input_img)
            input_label = torch.zeros(label.shape[0], label.shape[1], length, length)
            input_label[:, :, lefttop[0]:lefttop[0]+h, lefttop[1]:lefttop[1]+w] = label
            input_label = transform(input_label)
            input_img = input_img.cuda()
            pred = net(input_img)
            mse = 0
            for i in range(input_label.shape[1]):# 4种类别
                mse += criteria(pred[i].cpu(), input_label[:, i, :, :].unsqueeze(0))
            mse_adder(mse.item())
            print(image_name[0][:-4]+': MSE= %.2f' % mse)

            if args.save_image == True:
                pred0 = np.array(torch.clamp(pred[0].cpu(), 0, 1))[0][0]
                pred1 = np.array(torch.clamp(pred[1].cpu(), 0, 1))[0][0]
                pred2 = np.array(torch.clamp(pred[2].cpu(), 0, 1))[0][0]
                pred3 = np.array(torch.clamp(pred[3].cpu(), 0, 1))[0][0]
                name, suffix = os.path.splitext(image_name[0])
                suffix = '.jpg'
                mask0 = np.array(input_label[:, 0, :, :]).squeeze()
                mask1 = np.array(input_label[:, 1, :, :]).squeeze()
                mask2 = np.array(input_label[:, 2, :, :]).squeeze()
                mask3 = np.array(input_label[:, 3, :, :]).squeeze()
                cv2.imwrite(os.path.join(args.result_dir, name+'_mask0'+suffix), mask0/np.max(mask0)*255)
                cv2.imwrite(os.path.join(args.result_dir, name+'_mask1'+suffix), mask1/np.max(mask1)*255)
                cv2.imwrite(os.path.join(args.result_dir, name+'_mask2'+suffix), mask2/np.max(mask2)*255)
                cv2.imwrite(os.path.join(args.result_dir, name+'_mask3'+suffix), mask3/np.max(mask3)*255)

                cv2.imwrite(os.path.join(args.result_dir, name+'_pred0'+suffix), pred0/np.max(pred0)*255)
                cv2.imwrite(os.path.join(args.result_dir, name+'_pred1'+suffix), pred1/np.max(pred1)*255)
                cv2.imwrite(os.path.join(args.result_dir, name+'_pred2'+suffix), pred2/np.max(pred2)*255)
                cv2.imwrite(os.path.join(args.result_dir, name+'_pred3'+suffix), pred3/np.max(pred3)*255)

        print('=======================================')
        print('The average MSE is %.2f ' % (mse_adder.average()))

