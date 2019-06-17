
from __future__ import print_function, division

import pandas as pd
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import argparse

from PIL import Image
import matplotlib.pyplot as plt

from test import load_model

def prediction_bar(output):

    cwd = os.getcwd()
    labels = pd.read_csv(cwd+'/labels.csv')
    
    output = output.detach().numpy()
    pred_labels = output.argsort()[0]
    pred_labels = np.flip(pred_labels[-1*len(pred_labels):])
    
    prediction, label = [], []
    
    for i in pred_labels[:5]:
        prediction.append(float(output[:,i]*100))
        label.append(str(i))
        
    for i in pred_labels[:5]:
        print('Class: {} , confidence: {:.2f}%'.format(labels.iloc[int(i)].values,float(output[:,i])*100))
        
    plt.bar(label,prediction, color='green')
    plt.title("Confidence Score Plot")
    plt.xlabel("Confidence Score")
    plt.ylabel("Class number")
    plt.show()
    
    return None
 
def img_plot(image):
    
    plt.imshow(image)
    plt.show()

def predict_one_image(model,img_fname):

    data_transforms_test = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    image = Image. open(img_fname).convert('RGB')
    image_tr= data_transforms_test(image)

    data = image_tr.expand(1,-1,-1,-1)

    probs = nn.Softmax(dim = 1)
    output = model(data)
    output = probs(output)
    _, predicted = torch.max(output.data, 1)
    
    img_plot(image)
    prediction_bar(output)
    
    return predicted





def main(args):

    cwd = os.getcwd()
    model=load_model(cwd+'/resnext101_94.8.pth', 'cpu')
    model.eval()

    predict_one_image(model,cwd+"/"+args.image_fname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test images')

    parser.add_argument('--image_fname', required=True, type = str,
                        help='path to image')
    
    args = parser.parse_args()

    main(args)

