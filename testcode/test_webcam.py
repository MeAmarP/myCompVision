#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:16:52 2019

@author: amarp
"""

import cv2
import os
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('myVid.avi',fourcc,fps,(640,480))

if not cap.isOpened():
    raise IOError('Cannot Run WebCam')
    
while True:
    _,frame = cap.read()
    out.write(frame)
    plt.imshow(frame)
cap.release()
    
    