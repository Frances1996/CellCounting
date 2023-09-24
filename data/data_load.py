import torch
import numpy as np
import os
import json
import cv2
from torch.utils.data import Dataset, DataLoader
from xml.dom import minidom
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy import signal
import unittest
import openslide
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import random_split
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
import warnings
from random import choice
from .data_aug import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairCutout, PairCutmix, PairSplice, PairColorJitter



def get_kernel(kernel_size, alpha):
# kernel defination:Pixel-to-Pixel Learning With Weak Supervision for Single-Stage Nucleus Recognition in Ki67 Images
    kernel = np.zeros((2 * kernel_size + 1, 2 * kernel_size + 1))
    for i in range(2 * kernel_size):
        for j in range(2 * kernel_size):
            if np.sqrt((kernel_size-i)**2+(kernel_size-j)**2) < kernel_size:
                kernel[i, j] = (np.e**alpha*(1-np.sqrt((kernel_size-i)**2+(j-kernel_size)**2)/kernel_size)-1)/(np.e**alpha - 1)
            else:
                kernel[i, j] = 0
    kernel[kernel < 0] = 0
    return kernel

# class TestKernel(unittest.TestCase):
#     def setUp(self):
#         print("test case start")
#
#     def tearDown(self):
#         print("test case end")
#
#     def test_max(self):
#         kernel = get_kernel(25, 3)
#         self.assertEqual(np.max(kernel), 1)
#
#     def test_min(self):
#         kernel = get_kernel(25, 3)
#         self.assertEqual(np.min(kernel), 0)
        
def get_label(annotation_name, kernel, image_size, n_classes):
    # "Epithelial": 1
    # "Lymphocyte": 2
    # "Neutrophil": 3
    # "Macrophage": 4
    label = np.zeros((image_size[1], image_size[0], n_classes))
    tree = ET.parse(annotation_name)
    root = tree.getroot()
    kernel_size = int((kernel.shape[0]-1)/2)
    for k in range(n_classes):
        sub_label = np.zeros((image_size[1], image_size[0]))
        if k < len(root):
            for child in root[k]:
                for x in child:
                    r = x.tag
                    if r == 'Attribute':
                        symptom = x.attrib['Name']
                    if r == 'Region':
                        vertices = x[1]
                        coords = np.zeros((len(vertices), 2))
                        for i, vertex in enumerate(vertices):
                            coords[i][0] = vertex.attrib['X']
                            coords[i][1] = vertex.attrib['Y']
                        center = np.mean(coords, 0)
                        if int(np.round(center[1]))<image_size[1] and int(np.round(center[0]))<image_size[0]:
                            sub_label[int(np.round(center[1])), int(np.round(center[0]))] = 1
            sub_label = signal.convolve(sub_label, kernel)
            sub_label[sub_label < 0] = 0
            sub_label = sub_label[kernel_size:sub_label.shape[0]-kernel_size, kernel_size:sub_label.shape[1]-kernel_size]
            if symptom == 'Epithelial':
                label[:, :, 0] = sub_label
            if symptom == 'Lymphocyte':
                label[:, :, 1] = sub_label
            if symptom == 'Neutrophil':
                label[:, :, 2] = sub_label
            if symptom == 'Macrophage':
                label[:, :, 3] = sub_label
    return label

def random_split(dataset, lengths, generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    # >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    # >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    # ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


class cell_dataset(Dataset):
    def __init__(self, filedir, kernel_size, alpha, n_classes, transform, test=False):
        self.img_path = os.path.join(filedir, 'svs')
        self.anno_path = os.path.join(filedir, 'xml')
        self.img_list = os.listdir(self.img_path)
        self.anno_list = os.listdir(self.anno_path)
        self.transform = transform
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.n_classes = n_classes
        self.test = test

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_name = self.img_list[idx]
        image = openslide.OpenSlide(os.path.join(self.img_path, image_name))
        image_size = image.level_dimensions[0]
        image = np.array(image.read_region((0, 0), 0, image.level_dimensions[0]))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        image = np.float32(image)/255
        annotation_name = self.anno_list[idx]
        kernel = get_kernel(self.kernel_size, self.alpha)
        label = get_label(os.path.join(self.anno_path, annotation_name), kernel, image_size, self.n_classes)
        label = np.float32(label)
        if self.test:
            image, label = self.transform(image, label)
            return image, label, image_name
        else:
            image, label = self.transform[0](image, label)
            img2_name = choice(self.img_list)
            img2 = openslide.OpenSlide(os.path.join(self.img_path, img2_name))
            img2_size = img2.level_dimensions[0]
            img2 = np.array(img2.read_region((0, 0), 0, img2.level_dimensions[0]))
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGBA2BGR)
            label2_name = os.path.splitext(img2_name)[0] + '.xml'
            label2 = get_label(os.path.join(self.anno_path, label2_name), kernel, img2_size, self.n_classes)
            image, label = self.transform[1](image, label, img2, label2)
            return image, label[0, :, :].unsqueeze(0), label[1, :, :].unsqueeze(0), label[2, :, :].unsqueeze(0), label[3, :, :].unsqueeze(0)


def test_dataloader(path, kernel_size, alpha, n_classes, batch_size, num_workers):
    img_dir = os.path.join(path, 'MoNuSAC_Test')
    transform = PairToTensor()
    dataloader = DataLoader(
        cell_dataset(img_dir, kernel_size, alpha, n_classes, transform, test=True),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == '__main__':
    # unittest.main(exit=False)
    kernel_size = 15
    alpha = 5
    batch_size = 1
    num_workers = 0
    kernel = get_kernel(kernel_size, alpha)
    constant = np.sum(kernel)
    n_classes = 4
    img_size = 512
    path = r'G:\papers\TheCellCount(paper)\CellCounting\multiclass_dataset\MoNuSAC'
    img_dir = r'G:\papers\TheCellCount(paper)\CellCounting\multiclass_dataset\MoNuSAC\MoNuSAC_train'
    dataloader = test_dataloader(path, kernel_size, alpha, n_classes, img_size, batch_size=1, num_workers=0)
    # transform = [PairCompose([
    #     PairToTensor(),
    #     PairSplice(512),
    #     transforms.RandomChoice([
    #         PairRandomHorizontalFilp(),
    #         PairColorJitter(0.3, 0.3, 0.3, 0.2),
    #         PairCutout(2)])
    #         ]), PairCutmix(2)]
    # celldata = cell_dataset(img_dir, kernel_size, alpha, n_classes, transform)
    # train_dataset, val_dataset = random_split(
    #     dataset=celldata,
    #     lengths=[0.7, 0.3],
    #     generator=torch.Generator().manual_seed(0)
    # )
    # train_data = DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )
    for i, data in enumerate(dataloader):
        image, label0, label1, label2, label3, _ = data
        image = np.array(image)[0].transpose(1, 2, 0)
        l0 = np.array(label0)[0][0]
        l1 = np.array(label1)[0][0]
        l2 = np.array(label2)[0][0]
        l3 = np.array(label3)[0][0]
        l = l0 + l1 + l2 + l3
        l = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)
        a = cv2.addWeighted(l, 0.5, image, 0.5, 0)
        cv2.imshow('a', a)
        cv2.waitKey(0)