import cv2
import os
import numpy as np
import openslide
import xml.etree.ElementTree as ET
from skimage import io
from scipy import signal



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

img_path = r'E:\CellCounting\multiclass_dataset\MoNuSAC\MoNuSAC_Test\svs'
xml_path = r'E:\CellCounting\multiclass_dataset\MoNuSAC\MoNuSAC_Test\xml'
save_path = r'E:\CellCounting\multiclass_dataset\MoNuSAC\MoNuSAC_Test\visual'
image_list = os.listdir(img_path)
kernel_size = 15
alpha = 5
kernel = get_kernel(kernel_size, alpha)
kernel_size = int((kernel.shape[0]-1)/2)
for image_name in image_list:
    name, suffix = os.path.splitext(image_name)
    patient_path = os.path.join(save_path, name)
    if not os.path.exists(patient_path):
        os.makedirs(patient_path)

    image = openslide.OpenSlide(os.path.join(img_path, image_name))
    image_size = image.level_dimensions[0]
    image = np.array(image.read_region((0, 0), 0, image.level_dimensions[0]))
    image = np.float32(cv2.cvtColor(image, cv2.COLOR_RGBA2BGR))
    annotation_name = os.path.join(xml_path, name+'.xml')
    tree = ET.parse(annotation_name)
    root = tree.getroot()
    for subroot in root:
        sub_label = np.zeros((image_size[1], image_size[0]))
        for child in subroot:
            for x in child:
                r = x.tag
                if r == 'Attribute':
                    label = x.attrib['Name']
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
        sub_label = np.clip(sub_label, 0, 1)
        sub_label = 255*np.float32(sub_label[kernel_size:sub_label.shape[0]-kernel_size, kernel_size:sub_label.shape[1]-kernel_size])
        sub_label = cv2.cvtColor(sub_label, cv2.COLOR_GRAY2BGR)
        visual = cv2.addWeighted(sub_label, 0.5, image, 0.5, 0)
        cv2.imwrite(os.path.join(patient_path, label+'.jpg'), visual)