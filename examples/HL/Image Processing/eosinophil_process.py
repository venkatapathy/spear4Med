import cv2
import numpy as np
import os
import argparse

def contour_threshold(img, thresh = 0, isMask = False):
    if not isMask:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #get threshold image
    ret,thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
    #find contours
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

parser = argparse.ArgumentParser(description='Eosinophil')
parser.add_argument('--imagePath', default="./images/patch_position(63852_33420)_125_Eosinophil.jpg",
                    help='input image file name')
parser.add_argument('--outputDir', default="output",
                    help='output directory name')
args = parser.parse_args()

# Check if output directory exists, if not create it
if not os.path.exists(args.outputDir):
    os.mkdir(args.outputDir)

# Create Output Subdirectories
eosino = os.path.join(args.outputDir,"patch_eosinophil")
eosino2 = os.path.join(args.outputDir,"patch_eosi2")
if not os.path.exists(eosino):
    os.mkdir(eosino)
if not os.path.exists(eosino2):
    os.mkdir(eosino2)

# Check if image exists
if not os.path.exists(args.imagePath):
    raise Exception("The path to the input Image file does not exist")

"""Extracting First Color"""

def first_color(image):
    lower_purple = np.array([150, 70, 165])
    upper_purple = np.array([230, 110, 195])

    mask = cv2.inRange(image, lower_purple, upper_purple)
    result_1 = cv2.bitwise_and(image, image, mask=mask)
    test_eosi = result_1.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilated = cv2.dilate(test_eosi, kernel)
    contours = contour_threshold(dilated.copy(), 20)

    count = 0
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        image_2 = image.copy()
        
        if(w*h>300):
            cropped_image = image_2[y:y+h, x:x+w]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(eosino2,f'patch_thresh_eosi{count}.jpg'), cropped_image)
            count +=1

"""Eosinophil with Nucleolus"""

def eosino_with_nucleous(eosinophil,i):
    flag = False
    height = eosinophil.shape[0]
    width = eosinophil.shape[1]
    contours = contour_threshold(eosinophil, 90)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if(w*h>100 and w*h < height*width):
            eosinophil = cv2.rectangle(eosinophil,(x,y),(x+w,y+h),(0,255,0),2)
            flag = True
    if flag:
        cv2.imwrite(os.path.join(eosino,f'eosino_{i}.jpg'), eosinophil)

image = cv2.imread(args.imagePath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
first_color(image)
count = 0
for file in os.listdir(eosino2):
    eosinophil = cv2.imread(os.path.join(eosino2,file))
    eosinophil = cv2.cvtColor(eosinophil, cv2.COLOR_BGR2RGB)
    eosino_with_nucleous(eosinophil,count)
    count += 1
