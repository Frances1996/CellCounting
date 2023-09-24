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


def get_mask(dir, image_size):
    doc = minidom.parse(dir)
    vectices_list = doc.getElementsByTagName("Vertices")
    mask = np.zeros(image_size)
    point_annotation = np.zeros(image_size)
    ind = 1
    for vectices in vectices_list:
        vertex_list = vectices.getElementsByTagName("Vertex")
        data = [[0, 0] for _ in range(len(vertex_list))]
        for i in range(len(vertex_list)):
            vertex = vertex_list[i]
            point_x = float(vertex.getAttribute('X'))
            point_y = float(vertex.getAttribute('Y'))
            data[i][0] = point_x
            data[i][1] = point_y
            if int(np.around(point_x)) < 1000 and int(np.around(point_y)) < 1000:
                mask[int(np.around(point_y)), int(np.around(point_x))] = ind
        ind += 1
        center_y, center_x = np.mean(data, 0)
        point_annotation[int(np.around(center_x)), int(np.around(center_y))] = 1

    return mask, point_annotation

def get_kernel(kernel_size, alpha):
    kernel = np.zeros((2 * kernel_size + 1, 2 * kernel_size + 1))
    for i in range(2 * kernel_size):
        for j in range(2 * kernel_size):
            if np.sqrt((kernel_size-i)**2+(kernel_size-j)**2) < kernel_size:
                kernel[i, j] = (np.e**alpha*(1-np.sqrt((kernel_size-i)**2+(j-kernel_size)**2)/kernel_size)-1)/(np.e**alpha - 1)
            else:
                kernel[i, j] = 0
    kernel[kernel < 0] = 0
    return kernel


class TestKernel(unittest.TestCase):
    def setUp(self):
        print("test case start")
 
    def tearDown(self):
        print("test case end")

    def test_max(self):
        kernel = get_kernel(25, 3)
        self.assertEqual(np.max(kernel), 1)

    def test_min(self):
        kernel = get_kernel(25, 3)
        self.assertEqual(np.min(kernel), 0)
        

class cell_dataset(Dataset):
    def __init__(self, filedir, kernel_size, alpha, transform=None):
        self.image_filedir = os.path.join(filedir, 'images')
        self.anno_filedir = os.path.join(filedir, 'Annotations')
        self.img_list = os.listdir(self.image_filedir)
        self.anno_list = os.listdir(self.anno_filedir)
        self.transform = transform
        self.kernel_size = kernel_size
        self.alpha = alpha

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_name = self.img_list[idx]
        image = cv2.imread(os.path.join(self.image_filedir, image_name))
        anno_name = self.anno_list[idx]
        image_size = (np.shape(image)[0], np.shape(image)[1])
        kernel = get_kernel(self.kernel_size, self.alpha)
        mask, point_annotation = get_mask(os.path.join(self.anno_filedir, anno_name), image_size)
        prob_annotation = signal.convolve(kernel, point_annotation)
        prob_annotation = prob_annotation[self.kernel_size:self.kernel_size+np.shape(image)[0], self.kernel_size:self.kernel_size+np.shape(image)[1]]
        prob_annotation[prob_annotation < 0] = 0
        return mask, prob_annotation, point_annotation, image


if __name__ == '__main__':
    unittest.main(exit=False)
    kernel_size = 15
    alpha = 5
    kernel = get_kernel(kernel_size, alpha)
    constant = np.sum(kernel)

    path = r'E:\CellCounting\dataset\MoNuSegTrainData'
    CellDataset = cell_dataset(path, kernel_size, alpha)
    for i, data in enumerate(CellDataset):
        mask, prob_annotation, point_annotation, image = data
        prob_annotation = cv2.cvtColor(np.float32(prob_annotation), cv2.COLOR_GRAY2BGR)
        cv2.imshow('prob_annotation', cv2.addWeighted(np.uint8(prob_annotation*150), 0.5, image, 0.5, 0))
        cv2.waitKey(0)
        