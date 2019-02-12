#!/usr/bin/env python3
import cv2
import os
import numpy as np
from PIL import Image
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("frontalface_default.xml");

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)] 
   # print(imagePaths)
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        # converting image to grayscale
        PIL_img = Image.open(imagePath).convert('L')
        # converting PIL image to numpy array using array() method of numpy
        img_numpy = np.array(PIL_img,'uint8')
        # Getting the image id
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        name = os.path.split(imagePath)[-1].split(".")[0]
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return faceSamples,ids
# Getting
faces,ids = getImagesAndLabels('training_cp')
print(ids)
# Training 
recognizer.train(faces, np.array(ids))
# Saving the model 
assure_path_exists('saved_model/')
recognizer.save('saved_model/s_model.yml')
