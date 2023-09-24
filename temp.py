import os
import cv2
import numpy as np

img = cv2.imread(r'G:\papers\The_Dissertation\subchapters\4cellcounting\dataset\MBM_data\BM_GRAZ_HE_0001_01_001.png')
a = np.empty(shape=(600, 0, 3))
for i in range(2):
    a = np.concatenate(img, axis=1)
print()