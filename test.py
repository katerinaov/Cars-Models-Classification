
from __future__ import print_function, division

from scipy.io import loadmat

import pandas as pd
import numpy as np


import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

import time
import os
import argparse

import cv2
from PIL import Image

import matplotlib.pyplot as plt

from dataset import CarsDataset


def get_dataframe(file_name):

    if file_name[-3:]=='mat':
        loaded_mat = loadmat(file_name)
        data_train = [[row.flat[0] for row in line] for line in loaded_mat["annotations"][0]]
        columns_train = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "class", "fname"]
        df = pd.DataFrame(data_train, columns=columns_train)
        df['class'] = df['class']-1 

        return df
    else:

        df = pd.read_csv(file_name)
        df['class'] = df['class']-1 
        return df

def create_dataloder(labels_file, bboxes):

    data_transforms_test = transforms.Compose([
    transforms.Resize((500, 500)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    df = get_dataframe(labels_file)
    test_set = CarsDataset(datafolder='./test', bboxes = bboxes, df=df, transform=data_transforms_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, num_workers=2)

    return test_loader

def test_set_accuracy(model, testloader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: %d %%' % (
        100 * correct / total))
    


def load_model(model_file, device):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    model_ft = models.resnext50_32x4d(pretrained=True) 
    model_ft.fc = torch.nn.Linear(model_ft.fc.in_features, 196)
    
    if device=='cpu':
        checkpoint = torch.load(model_file, map_location='cpu')
    else:
        checkpoint = torch.load(model_file)
    model_ft.load_state_dict(checkpoint)

    return model_ft
    
def main(args):


    test_loader = create_dataloder(args.labels_fname, args.bboxes)

    cwd = os.getcwd()
    model=load_model(cwd+'/resnext50_93.4.pth', args.device)
    model.eval()

    test_set_accuracy(model,test_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test images')

    parser.add_argument('--labels_fname', required=True, type = str,
                        help='path to file with labels (mat or csv')
    parser.add_argument('--device', required=True,  type=str,
                        help='cuda or cpu')
    parser.add_argument('--bboxes', required=True,  type=bool,
                        help='True or False in case upi brovide bounding boxes in labels file for images')


    args = parser.parse_args()

    main(args)

