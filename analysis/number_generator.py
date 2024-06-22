#!/usr/bin/env python

# the model will consist of synthetic images sourced from MNIST
# the UNIPEN library and a backdrop of table lines
# in many ways the HTR problem of detecting numbers in a table
# of climate data is a simple captcha problem with only a
# limited number of possible outcomes i.e. digits and a
# number of punctuation sets (.,-)
# using only this reduced set, instead of using a full blown
# HTR text recognition model (including all alpha numberic
# characters) should increase model robustness and compactness.

import random
import cv2
import numpy as np
import albumentations as A
import torch
from torchvision import datasets, transforms
from utils import *

# load the UNIPEN decimals
# TODO

# read in the background grid image and apply
# the random transform to the data
background = cv2.imread("grid_background.jpg")
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
background = transform_grid(image=background)['image']
height, width, _ = background.shape
blank_image = np.ones((height, width, 2), np.uint8) * 255
numbers = np.ones((height, width), np.uint8) * 255

# Define the transformation to convert images to PyTorch tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset with the specified transformation
mnist = datasets.MNIST(
  root='./dataset',
  train=True,
  download=True,
  transform=transform
  )

# Create a DataLoader to load the dataset in batches
train_loader_pytorch = torch.utils.data.DataLoader(
  mnist,
  batch_size=1,
  shuffle=True
  )

# build the number backwards from the last decimal to the decimal point
# onward to the minus sign
digits = []
include_decimal = True
include_minus = True

# Print the first few images in a random batch
for i, (image, label) in enumerate(train_loader_pytorch):
    if i < 2:  # Print the first 5 samples
      
        # grab image
        img = image[0].squeeze().numpy()
        img = 255 - (img * 255)
        
        # transform
        img = transform_number(image = img)['image']
        x, y = img.shape
        
        tmp = blank_image
        tmp[:,:, 0] = numbers
        start_x = (width - (i + 1) * 35)
        print(start_x)
        
        tmp[60:(60 + y), start_x - x:start_x, 1] = img
        numbers = np.min(tmp, axis=2)
        
        # save label item
        digits.insert(0, label.item())
        
        # insert decimal point, randomly
        if i == 0 and include_decimal and random.random() < 0.5:
          decimal_separator = random.choice([".", ","])
          digits.insert(0, decimal_separator)
          
          # grab random decimal character
          
          # transform
          
          # insert in numbers layer
          
    else:
        break  # Exit the loop after printing 5 samples

digits = [str(x) for x in digits]
value = ''.join(digits)

# add minus sign if desired
if include_minus and random.random() < 0.5:
    digits.insert(0, "-")
    value = "-" + value

background[:,:,1] = numbers
dst = np.min(background, axis=2)
dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

# final transformation
dst = transform_image(image = dst)['image']

# check conversion of formats
cv2.imwrite("test.png", dst)
print(digits)
