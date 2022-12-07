import h5py
from PIL import Image
import cv2
import torch
import numpy as np
import torch.nn as nn
import argparse
from torch.autograd import Variable
from skimage import segmentation
from skimage import morphology
from skimage import color
import torch.nn.functional as F
import torch.optim as optim
from skimage import data
from skimage import io
from skimage.color import rgb2hed, hed2rgb,rgb2gray

from skimage.filters import threshold_mean
import matplotlib.pyplot as plt
import matplotlib
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity
import numpy as np
import random
import csv
from skimage import measure
from skimage import feature
from math import sqrt



debug=False
use_cuda = torch.cuda.is_available()

#open the ds file
ds = h5py.File('/home/venkat/Projects/workbook/spear4Med/examples/PathDetection/data/training.h5', 'r')

#get a random list of 50 patches
randomlist = random.sample(range(1, 20000), 50)


# results of the labeling functions
f = open('/home/venkat/Projects/workbook/spear4Med/examples/PathDetection/output/test/test1.csv', 'w')

#roi = open('/home/venkat/Projects/workbook/spear4Med/examples/PathDetection/output/test/roi.csv', 'w')

# create the csv writer
writer = csv.writer(f)

row=["sourcepatchno", "bbox", "regionDetector"]

writer.writerow(row)

for patchno in randomlist:
    ihc_hed = rgb2hed(ds['x'][patchno])
    null = np.zeros_like(ihc_hed[:, :, 0])
    #dab
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    #esoin
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1],null), axis=-1))
    #hematoxylin
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0],null,null), axis=-1))

    #dab based score
    #mask based on dab
    ihc_d_gray=rgb2gray(ihc_d)
    thresh = threshold_otsu(ihc_d_gray)
    dabmask = ihc_d_gray < thresh
    dabmask=morphology.remove_small_holes(dabmask,10)
    dabmask=morphology.remove_small_objects(dabmask,20)
    #find individual object
    dablabels=measure.label(dabmask)
    dabregions=measure.regionprops(dablabels)

    dabbasedcount=0
    if debug:
        plt.imshow(ihc_h)
        plt.imshow(dabmask,alpha=0.4)
        print('Number of lympocytes',ds['y'][patchno])

    for region in dabregions:
        if region.equivalent_diameter_area > 20 and region.equivalent_diameter_area < 35:
            if debug:
                print("Diameter:%d"%(region.equivalent_diameter_area))
                plt.plot(region.centroid[1],region.centroid[0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
                plt.waitforbuttonpress()
            dabbasedcount+=1
    
    if debug:
        plt.close()

    #hematoxylin based score
    ihc_h_gray=rgb2gray(ihc_h)
    thresh = threshold_otsu(ihc_h_gray)
    hmask = ihc_h_gray < thresh
    hmask=morphology.remove_small_holes(hmask,10)
    hmask=morphology.remove_small_objects(hmask,20)
    #find individual object
    hlabels=measure.label(hmask)
    hregions=measure.regionprops(hlabels)

    hbasedcount=0

    if debug:
        plt.imshow(ihc_h)
        plt.imshow(hmask,alpha=0.4)
        print('Number of lympocytes',ds['y'][patchno])

    for region in hregions:
        if region.equivalent_diameter_area > 20 and region.equivalent_diameter_area < 35:
            
            #check if dab region is also present
            min_row, min_col, max_row, max_col = region.bbox

            for dab_region in dabregions:
                if(dab_region.centroid[0]>min_row and dab_region.centroid[0]<max_row) \
                 and (dab_region.centroid[1]>min_col and dab_region.centroid[1]<max_col):
                    if debug:
                        plt.plot(dab_region.centroid[1],dab_region.centroid[0], marker="o", markersize=5, markeredgecolor="pink", markerfacecolor="orange")
                        plt.waitforbuttonpress()
                    hbasedcount+=1
                    break
    if debug:    
        plt.close()
    
     #image based score
    ds_gray=rgb2gray(ds['x'][patchno])
    thresh = threshold_otsu(ds_gray)
    dsmask = ds_gray < thresh
    #dsmask=morphology.remove_small_holes(dsmask,10)
    #dsmask=morphology.remove_small_objects(dsmask,20)
    #find individual object
    dslabels=measure.label(dsmask)
    dsregions=measure.regionprops(dslabels)

    dsbasedcount=0

    if debug:
        plt.imshow(ds['x'][patchno])
        plt.imshow(dsmask,alpha=0.4)
        print('Number of lympocytes',ds['y'][patchno])

    for region in dsregions:
        if region.equivalent_diameter_area > 20 and region.equivalent_diameter_area < 35:
            
            #check if dab region is also present
            min_row, min_col, max_row, max_col = region.bbox

            for dab_region in dabregions:
                if(dab_region.centroid[0]>min_row and dab_region.centroid[0]<max_row) \
                 and (dab_region.centroid[1]>min_col and dab_region.centroid[1]<max_col):
                    if debug:
                        plt.plot(dab_region.centroid[1],dab_region.centroid[0], marker="o", markersize=5, markeredgecolor="pink", markerfacecolor="orange")
                        plt.waitforbuttonpress()
                    dsbasedcount+=1
                    break
    if debug:    
        plt.close()

    row=[patchno, ds['organ'][patchno],ds['y'][patchno], dabbasedcount,hbasedcount,dsbasedcount]

    writer.writerow(row)

# close the file
f.close()
