# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 23:47:56 2022

@author: User
"""
#"E:\\ml\\feature extraction\\python_code\\sir\\Dataset\\Valid\\Bacterial blight\\*.jpg

import cv2
import numpy as np
import pandas as pd
#from google.colab.patches import cv2_imshow
from skimage.color import rgb2gray
from skimage.feature import greycomatrix
from skimage import feature, io
from sklearn import preprocessing
from matplotlib import pyplot as plt
import glob

path =glob.glob("F:/internship/real time project/omden uae project/train/Subsidence/*.jpg")
#Dataset/Valid

for file in path:
    #print(file)
    img=cv2.imread(file,cv2.IMREAD_UNCHANGED)
   
    b,g,r = cv2.split(img)
    rgb_img = cv2.merge([r,g,b])
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((2,2),np.uint8)
    #opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(closing,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)
    # Threshold
    ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    plt.plot(211),plt.imshow(rgb_img)
    plt.savefig('org.png')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.plot(212),plt.imshow(thresh, 'gray')
    plt.savefig('gray.png')
    plt.imsave(r'thresh.png',thresh)
    plt.title("Otsu's binary threshold"), plt.xticks([]), plt.yticks([])
    plt.tight_layout()
    plt.show()
    img = cv2.imread(file,0)
    S = preprocessing.MinMaxScaler((0,11)).fit_transform(img).astype(int)
    Grauwertmatrix = feature.greycomatrix(S, [1,2,3], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=12, symmetric=False, normed=True)
    
    ContrastStats = feature.greycoprops(Grauwertmatrix, 'contrast')
    CorrelationtStats = feature.greycoprops(Grauwertmatrix, 'correlation')
    HomogeneityStats = feature.greycoprops(Grauwertmatrix, 'homogeneity')
    ASMStats = feature.greycoprops(Grauwertmatrix, 'ASM')
    
    arrcontrrast=np.mean(ContrastStats)
    arrcorrelation=np.mean(CorrelationtStats)
    arrasms=np.mean(ASMStats)
    arrhomo=np.mean(HomogeneityStats)
    print(arrcontrrast,arrcorrelation,arrhomo,arrasms)

    
    df=pd.DataFrame([arrcontrrast,arrcorrelation,arrasms,arrhomo]).T
    df.to_csv("F:/internship/real time project/omden uae project/train/testData.csv",mode='a',header=False)
    df.to_csv('my_csv.csv', mode='a', header=False)

    #cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    