import cv2


# Pretrained classes in the model
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value


# Loading model
model = cv2.dnn.readNetFromCaffe('models/caffe/MobileNetSSD_deploy.prototxt',
                                 'models/caffe/MobileNetSSD_deploy.caffemodel')
image = cv2.imread("img/workplace.jpg")
CenterVal = int(image.shape[0] / 2)
LowerCenterVal = int(CenterVal - (CenterVal*0.3)) #20Percent Below Center in HoriDir
UpperCenterVal = int(CenterVal + (CenterVal*0.3)) #20Percent Above Center in HoriDir
resize_image = cv2.resize(image,(300,300))
cols = resize_image.shape[1]
rows = resize_image.shape[0]
#image_height, image_width, _ = resize_image.shape

model.setInput(cv2.dnn.blobFromImage(resize_image,0.007843, size=(300, 300),mean=(127.5,127.5,127.5),swapRB=True))
output = model.forward()
# print(output[0,0,:,:].shape)


for detection in output[0, 0, :, :]:
    confidence = detection[2]
    class_id = int(detection[1])
    
    if confidence > .90 and class_id == 15:
        #class_name=id_class_name(class_id,classNames)
        #print(str(str(class_id) + " " + str(detection[2])  + " " + class_name))
        
        xLeftBot = int(detection[3] * cols)
        yLeftBot = int(detection[4] * rows)
        xRightTop = int(detection[5] * cols)
        yRightTop = int(detection[6] * rows)
        
        HeightFactor = image.shape[0]/300.0
        WidthFactor = image.shape[1]/300.0
        
        #scale BBox Image
        xLeftBot = int(WidthFactor * xLeftBot)
        yLeftBot = int(HeightFactor * yLeftBot)
        xRightTop = int(WidthFactor * xRightTop)
        yRightTop = int(HeightFactor * yRightTop)
        
        xCenter = (xLeftBot+xRightTop)/2
        yCenter = (yLeftBot+yRightTop)/2
                    
        if xCenter < LowerCenterVal:
            #Move Left
            cv2.putText(image,'L',(int(xCenter),int(yCenter)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2) 
            #motoClass.turnleft(motoClass.Turn10DutyCycle,motoClass.Turn10Time)
            #motoClass.stop()
            print('L')
        elif xCenter > UpperCenterVal:
            #Move Right
            cv2.putText(image,'R',(int(xCenter),int(yCenter)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            #motoClass.turnright(motoClass.Turn10DutyCycle,motoClass.Turn10Time)
            #motoClass.stop()
            print('R')
        elif xCenter > LowerCenterVal and xCenter < UpperCenterVal:
            cv2.putText(image,'C',(int(xCenter),int(yCenter)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            print('C')
            continue        
        #cv2.rectangle(image, (xLeftBot, yLeftBot), (xRightTop, yRightTop), (0,255,0), thickness=2)
cv2.imshow('Result Image', image)
#cv2.imwrite("resultImage.jpg",image)

