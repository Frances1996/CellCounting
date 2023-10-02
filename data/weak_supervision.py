import numpy as np
import cv2
import os
import json


path = r'G:\papers\TheCellCount(paper)\CellCounting\multiclass_dataset\MoNuSAC\MoNuSAC_train\visual'
patients = os.listdir(path)
for patient in patients:
    patient_path = os.path.join(path, patient)
    diseases = os.listdir(patient_path)
    for disease in diseases:
        name, suffix = os.path.splitext(disease)
        save_name = name + '_mask'
        if suffix == '.json':
            with open(os.path.join(patient_path, disease)) as json_file:
                data = json.load(json_file)
                instances = data['shapes']
                image_height = data['imageHeight']
                image_width = data['imageWidth']
                mask = np.zeros((image_height, image_width))
                for instance in instances:
                    corner_points = []
                    instance_label = instance['label']
                    points = instance['points']
                    for point in points:
                        x = int(round(point[0]))
                        y = int(round(point[1]))
                        corner_points.append((x, y))
                    cv2.fillPoly(mask, [np.array(corner_points)], (255, 255, 255))
                cv2.imwrite(os.path.join(patient_path, save_name+'.jpg'), mask)

