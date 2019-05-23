# -*- coding: utf-8 -*-
"""
Created on Thu May 23 20:43:51 2019

@author: apotdar
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcparams['axes.titlesize'] = 8
import glob
import cv2

base_dir_path = 'E:/Py Proj/ML/EXPLORES/deep_object_detect/fruits-360/'
train_dir_path = os.path.join(base_dir_path,'train')
test_dir_path = os.path.join(base_dir_path,'test')

def getAllClassNames(dir_path):
    return os.listdir(dir_path)

def understandData(BASE_DIR_PATH):    
    train_dir_path = os.path.join(BASE_DIR_PATH,'train')
    #test_dir_path = os.path.join(BASE_DIR_PATH,'test')
    print("Number of Classes = ",len(os.listdir(train_dir_path)))
    AllClassNames = os.listdir(train_dir_path)
    #print("Class Names = ",AllClassNames)
#    print('CLASS NAME'+'\t'+'NUMBER OF IMAGES')    
#    for class_name in AllClassNames:
#        print(class_name+'\t',len(os.listdir(os.path.join(train_dir_path,class_name))))
    displaySampleImages(train_dir_path,AllClassNames)
    return


def displaySampleImages(PATH_TO_TRAIN_DIR,ALL_CLASS_NAMES):
    #NoOfClasses = len(ALL_CLASS_NAMES)   
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.7, wspace=0.1)
    fig.suptitle('Understanding Fruit-360 Dataset', fontsize=16)
    for n,class_name in enumerate(ALL_CLASS_NAMES):
        ImagePath = glob.glob(os.path.join(PATH_TO_TRAIN_DIR,class_name)+'/*.jpg')[0]
        #print(ImagePath)
        Img = cv2.imread(ImagePath)
        ax = fig.add_subplot(10,10,(n+1))
        plt.imshow(cv2.cvtColor(Img, cv2.COLOR_BGR2RGB))
        ax.set_title(class_name+str(n))
        plt.axis('off')
    plt.show()
    return