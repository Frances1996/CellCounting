from xml.dom import minidom
import os
import numpy as np
import cv2


xml_path = r'G:\papers\TheCellCount(paper)\CellCounting\dataset\MoNuSegTrainData\Annotations'
img_path = r'G:\papers\TheCellCount(paper)\CellCounting\dataset\MoNuSegTrainData\images'
save_path = r'G:\papers\TheCellCount(paper)\CellCounting\dataset\MoNuSegTrainData\label'
filelist = os.listdir(xml_path)
for filename in filelist:
    name, suffix = os.path.splitext(filename)
    img = cv2.imread(os.path.join(img_path, name+'.tif'))
    doc = minidom.parse(os.path.join(xml_path, name+'.xml'))
    vectices_list = doc.getElementsByTagName("Vertices")
    data_list = []
    for vectices in vectices_list:
        vertex_list = vectices.getElementsByTagName("Vertex")
        this_data = [[0,0] for _ in range(len(vertex_list))]
        for i in range(len(vertex_list)):
            vertex = vertex_list[i]
            this_data[i][0] = float(vertex.getAttribute("X"))
            this_data[i][1] = float(vertex.getAttribute("Y"))
        data_list.append(this_data)
    masks = []
    for data in data_list:
        data = np.array(data)
        center = np.mean(data, 0)
        submask = np.zeros((1000, 1000)).astype(np.float32)
        submask[np.int(np.round(center[1])), np.int(np.round(center[0]))] = 1
        mask_filter_temp = cv2.GaussianBlur(submask, (25, 25), 0, 0, borderType=cv2.BORDER_CONSTANT)
        masks.append(mask_filter_temp)
    masks = np.array(masks)
    mask = np.max(masks, axis=0)
    mask = np.uint8(mask/np.max(mask)*255)
    cv2.imwrite(os.path.join(save_path, name + ".jpg"), mask)

