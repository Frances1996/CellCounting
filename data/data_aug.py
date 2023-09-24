import random
from typing import Any
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import cv2
import torch


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairToTensor(transforms.ToTensor):
    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
    

class PairCutout(object):
    def __init__(self, n_holes):
        self.n_holes = n_holes

    def __call__(self, img, label):
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        length = int(np.round(min(h, w))/5)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length//2, 0, h)
            y2 = np.clip(y + length//2, 0, h)
            x1 = np.clip(x - length//2, 0, w)
            x2 = np.clip(x + length//2, 0, w)
            mask[y1:y2, x1:x2] = 0
        mask = mask[:, :, np.newaxis]
        mask_image = np.repeat(mask, c, axis=2).transpose(2,0,1)
        out_img = img * mask_image
        c_label = label.shape[0]
        mask_label = np.repeat(mask, c_label, axis=2).transpose(2,0,1)
        out_label = label * mask_label
        return out_img, out_label

class PairCutmix(object):
    # 将img2上的patch贴在img1上
    def __init__(self, n_holes):
        self.n_holes = n_holes
    def __call__(self, img1, label1, img2, label2):
        transform = PairToTensor()
        img2, label2 = transform(img2, label2)
        c, h1, w1 = img1.shape[0], img1.shape[1], img1.shape[2]
        h2, w2 = img2.shape[1], img2.shape[2]
        h = min(h1, h2)
        w = min(w1, w2)
        length = int(np.round(min(h, w))/2)
        out = img1.clone()
        out_label = label1.clone()
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length//2, 0, h)
            y2 = np.clip(y + length//2, 0, h)
            x1 = np.clip(x - length//2, 0, w)
            x2 = np.clip(x + length//2, 0, w)
            out[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
            out_label[:, y1:y2, x1:x2] = label2[:, y1:y2, x1:x2]
        return out, out_label


class PairColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, img, label):
        transform = transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        out = transform(img)
        return out, label

class PairSplice(object):
    # 将图裁剪并拼接在一起成为固定size的图片,图像变换，没有扩增
    def __init__(self, img_size):
        self.img_size = img_size
    def __call__(self, img, label):
        w = min(img.shape[1], img.shape[2])
        h = max(img.shape[1], img.shape[2])
        transform = PairRandomCrop(self.img_size)
        if w >= self.img_size:
            out, out_label = transform(img, label)
        elif h <= self.img_size:
            radio1 = int(np.ceil(self.img_size/h))
            radio2 = int(np.ceil(self.img_size/w))
            if h == img.shape[1]:
                out = torch.zeros((img.shape[0], h*radio1, w*radio2))
                out_label = torch.zeros((label.shape[0], h*radio1, w*radio2))
                for i in range(radio1):
                    for j in range(radio2):
                        out[:, i*h:i*h+h, j*w:j*w+w] = img
                        out_label[:, i*h:i*h+h, j*w:j*w+w] = label
            else:
                out = torch.zeros((img.shape[0], w*radio2, h*radio1))
                out_label = torch.zeros((label.shape[0], w*radio2, h*radio1))
                for i in range(radio2):
                    for j in range(radio1):
                        out[:, i*w:i*w+w, j*h:j*h+h] = img
                        out_label[:, i*w:i*w+w, j*h:j*h+h] = label
            out, out_label = transform(out, out_label)
        else:
            radio = int(np.ceil(self.img_size/w))
            if h == img.shape[1]:
                out = torch.zeros((img.shape[0], h, w*radio))
                out_label = torch.zeros((label.shape[0], h, w*radio))
                for i in range(radio):
                    out[:, :, i*w:i*w+w] = img
                    out_label[:, :, i*w:i*w+w] = label
            else:
                out = torch.zeros((img.shape[0], w*radio, h))
                out_label = torch.zeros((label.shape[0], w*radio, h))
                for i in range(radio):
                    out[:, i*w:i*w+w, :] = img
                    out_label[:, i*w:i*w+w, :] = label
            out, out_label = transform(out, out_label)
        return out, out_label