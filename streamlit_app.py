#%%writefile cvcheck.py
import cv2
import torch
from torchvision import datasets, models, transforms
import streamlit as st
from imutils import paths
import numpy as np
import imutils
import torchvision
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from PIL import Image

class_names = ['with_mask',
 'without_mask'
]
maskNet = torch.load('mask_detection_model_resnet101.pth',map_location=torch.device('cpu'))

Known_distance = 24.0
Known_width = 14.3

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

fonts = cv2.FONT_HERSHEY_COMPLEX

def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key = cv2.contourArea)
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

bbox = ''
count = 0
colors = np.random.uniform(0, 255, size=(2, 3))
#font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2

def process_image(image):
    
    pil_image = image
   
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = image_transforms(pil_image)
    return img


def classify_face(image):
    device = torch.device("cpu")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #im_pil = image.fromarray(image)
    #image = np.asarray(im)
    im = Image.fromarray(image)
    image = process_image(im)
    st.write('image_processed')
    img = image.unsqueeze_(0)
    img = image.float()

    maskNet.eval()
    maskNet.cpu()
    output = maskNet(image)
    st.write(output,'##############output###########')
    out, predicted = torch.max(output, 1)
    st.write(predicted.data[0],"predicted")


    classification1 = predicted.data[0]
    index = int(classification1)
    st.write(class_names[index])
    return class_names[index]

st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = frame.shape
    outputs = classify_face(frame)
    marker = find_marker(frame)
    focalLength = (marker[1][0] * Known_distance) / Known_width
    inches = distance_to_camera(Known_width, focalLength, marker[1][0])
      # draw a bounding box around the image and display it
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)
    #cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
    cv2.putText(frame, "%.2fft" % (inches / 12),
        (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        2.0, (0, 255, 0), 3)
    cv2.putText(frame,str(outputs),(100,height-20), fonts, 1,(255,255,255),1,cv2.LINE_AA)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')
