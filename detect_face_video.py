import cv2
import numpy as np
import matplotlib.pyplot as plt


#prototxt_path = 'deploy.prototxt.txt'
config_path = 'resnet_10_fp16/deploy.protoxt'
model_path = 'resnet_10_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel'

#load the model
#faceNet = cv2.dnn.readNetFromTensorflow(model_path,config_path)
faceNet = cv2.dnn.readNetFromCaffe(config_path,model_path)

#read image file
capVid = cv2.VideoCapture('meamar.mp4')
frame_width = int(capVid.get(3))
frame_height = int(capVid.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (frame_width,frame_height))



while capVid.isOpened():
    ret,frame = capVid.read()
    if ret == True:
        (h,w) = frame.shape[:2]
        resizeMainImg = cv2.resize(frame,(300,300))
        #frame pre-processing
        blob = cv2.dnn.blobFromImage(resizeMainImg,1.0,(300,300))

        faceNet.setInput(blob)
        detects = faceNet.forward()

        for obj in range(0,detects.shape[2]):
            confidence = detects[0,0,obj,2]
            
            if confidence > 0.55:
                box = detects[0,0,obj,3:7] * np.array([w,h,w,h])
                (x0,y0,xend,yend)=box.astype("int")
                cv2.rectangle(frame,(x0,y0),(xend,yend),(127,255,0),2)
        out.write(frame)
    else:
        break

capVid.release()
out.release()
#output.release()
        
