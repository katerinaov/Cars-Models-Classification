# Summary
This is a project for [Grab competition AI for S.E.A](https://www.aiforsea.com/computer-vision). The task was to create a
Data Model based on the Stanford "Cars" dataset in order to solve the problem statement.
The final result for classifier - 94.81% accuracy.

# Approach
It's a Computer Vision task, so nowadays Transfer Learning is showing the best results in Image Recognition problems. So I decided to proceed with this approach. My plan was:
* try different models, such as resnet(34, 50, 101, 152), densenet(121, 161, 169, 201), vgg (16,19), inception v3, resnext
* try various image transformations from sizing to brightening modifications
* ensemble technique for best-perfomed models

I worked with PyTorch library and completed all the training on Free GPU (Tesla K80) provided by Google in Google Colaboratory.

# Dataset

The dataset provided by Stanford, it's available [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). 

The dataset consists of 2 sets: training and testing, with 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe. Also, all the classes names and bounding boxes for images provided in .mat format.

# Implementation

For image transformations, I used PyTorch 'transforms' method and 'albumentations' [library](https://github.com/albu/albumentations) (it's new, fast, and as a result showed better results).
I tried different parameters and types of image transformations, but come up with next 2 structures:
* Pytorch: Resize(), RandomHorizontalFlip(), RandomRotation(15), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
* Albumentations: Resize(500, 500),HorizontalFlip(),RandomBrightnessContrast(), ShiftScaleRotate(rotate_limit=15, scale_limit=0.10), JpegCompression(80), HueSaturationValue(), Normalize(),

| Model | Image Transforms | Accuracy Score |
| ------ | ------ |------ |
|ResNet34| pytorch| 87.88 |
|ResNet101|albumenatations | 93.07|
|ResNet101| pytorch| 90.70|
|ResNet152|albumenatations | 92.67 |
|Densenet121| albumenatations|89.71 |
|Densenet201|albumenatations |87.53 |
|ResNext50_32x4d|albumenatations |93.39|
|ResNext100_32x8d|albumenatations |94.81|

# How to test

To test the model, you'll need to download this model from [here](https://drive.google.com/open?id=1--BqZCxQog_6mpVjNIu0QS-tY944OL-S93)

There 2 ways to test my model:
* by testing only one image (get predicted class and confidence)
* testing one dataset

1) You need to load model .pth file, labels.csv, test.py and test_one_image.py scripts and save it in directory like:
     * image file to test
     * test.py
     * test_one_image.py
     * model.pth
    *  labels.csv
```sh
$ python3 test_one_image.py --image_fname 00001.jpg
```

2) You need to load model pth file, test.py and dataset.py scripts and save it in derictory like:
     * test (folder_with_test_images)
      -image1
      -image2
      -image3
      -...
     * test.py
     * dataset.py
     * model.pth
    *  labels_file(mat or csv format)

```sh
$ python3 test.py --labels_fname test_labels.csv --device cpu --bboxes True
```
Here you can choose whether you want to run the script using cuda or cpu.
Also you need to add whether you're providing bounding boxes for images in the test_labels file(it's just example, you'll need to change a name of a file), which can be in .mat or .csv format.

# Requirements

* pytorch == 1.1.0
* torch.vision == 0.3.0
* numpy
* pandas
* PIL
* cv2
* Matplotlib

# Conclusion
The model - [Resnext](https://arxiv.org/abs/1611.05431) (101_32x8d) showed the best result - 94.81% accuracy along with image transfomations with albumentations library on the image size of 400. Moreover, image tansformation using structer presented with albumentations library always showed result for 1-1.5% better comparing to PyTorch transforms methods. As you can see in the Notebook (cars_classifier.ipynb) while training model hasn't been tending to overfit, as you can check by plots of loss and accuracy. Overall, this work showed that with even limited computer power (I've done all the work on MacBook Air 2013 and Google Colab) and time resources, we can build easy architecture using transfer learning with PyTorch, which reached quite a good result.

# Notes

I was planning to try ensemble technique to combine several models using weights for each model, but couldn't complete it on time due to lack of computing power (All the work is done in Google Colab). Sometimes, this technique works well with such task and might add 1-2% to final accuracy. So, this is my next step in Stanford Cars Dataset classification problem. Also, some further hyperparameter tuning might be also helpful.
