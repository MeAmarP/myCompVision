import cv2
import numpy as np


#prototxt_path = 'deploy.prototxt.txt'
config_path = 'resnet_10_fp16/deploy.protoxt'
model_path = 'resnet_10_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel'

#load the model
#faceNet = cv2.dnn.readNetFromTensorflow(model_path,config_path)
faceNet = cv2.dnn.readNetFromCaffe(config_path,model_path)

#read image file
MainImg = cv2.imread('img/meamar.jpg')
(h,w) = MainImg.shape[:2]
resizeMainImg = cv2.resize(MainImg,(300,300))
#image pre-processing
blob = cv2.dnn.blobFromImage(resizeMainImg,1.0,(300,300))

faceNet.setInput(blob)
detects = faceNet.forward()

for obj in range(0,detects.shape[2]):
    confidence = detects[0,0,obj,2]
    
    if confidence > 0.55:
        box = detects[0,0,obj,3:7] * np.array([w,h,w,h])
        (x0,y0,xend,yend)=box.astype("int")
        cv2.rectangle(MainImg,(x0,y0),(xend,yend),(127,255,0),2)
        
#cv2.imshow("Output",MainImg)
cv2.imwrite('img/meamar_detect.jpg',MainImg)