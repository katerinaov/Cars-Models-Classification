
from PIL import Image
import cv2
import os

import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CarsDataset(Dataset):
    def __init__(self, datafolder, bboxes, df=None, transform = transforms.Compose([transforms.ToTensor()])):
        
        self.datafolder = datafolder
        self.df = df
        self.labels = df['class'].values
        self.image_fname = df['fname'].values
        self.bboxes = bboxes
        self.image_files_list = [s for s in os.listdir(datafolder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_files_list)
    
    def crop_bbox(self,bbox,img_fname):
        """ Method cropps image by provided in .mat file bounding boxes
        """
        image = cv2.imread(img_fname)
        l1, l0,_ = image.shape
        b0 = bbox[2] - bbox[0]
        b1 = bbox[3] - bbox[1]
        x0n,x1n = max(int(bbox[0] - b0*0.05),0), min(int(bbox[2] + b0*0.05),l0-1)
        y0n,y1n = max(int(bbox[1] - b1*0.05),0), min(int(bbox[3] + b1*0.05),l1-1)
            
        image = image[y0n:y1n,x0n:x1n]
        return  image
    
    
    def __getitem__(self, idx):
        
        img_name = os.path.join(self.datafolder, self.image_fname[idx])
        label = self.labels[idx]

        if self.bboxes:
            bboxes_arr = self.df[['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values
            image = self.crop_bbox(bboxes_arr[idx], img_name)
        else:
            image = mage = cv2.imread(img_name)
        
        image = Image. fromarray(image).convert('RGB')
        image = self.transform(image)

        return image, label
