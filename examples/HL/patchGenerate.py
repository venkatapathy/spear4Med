# The path can also be read from a config file, etc.

from config import *

import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

import time
import cv2
import shutil
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator


def generate_all_patches():
    image = WholeSlideImage(PATH)
    dimensions = image.get_shape_from_spacing(SPACING)

    print(dimensions[0], dimensions[1])
    try:
        os.mkdir(PATCH_DIRECTORY)
        print("Directory " , PATCH_DIRECTORY ,  " Created ") 
    except FileExistsError:
        # shutil.rmtree(PATCH_DIRECTORY)
        print("Directory " , PATCH_DIRECTORY ,  " already exists")
        # os.mkdir(PATCH_DIRECTORY)

    t1 = time.time()
    count = 0
    for startY in range(0, dimensions[1], PATCH_HEIGHT):
        for startX in range(0, dimensions[0], PATCH_WIDTH):
            count+=1
            patch = image.get_patch(startX, startY, PATCH_WIDTH, PATCH_HEIGHT, SPACING)
            cv2.imwrite(PATCH_DIRECTORY + f'/patch_{count}.jpg', patch)
    print(f'Total Time: {time.time() - t1}')


def generate_single_patch():
    image = WholeSlideImage(PATH)
    patch = image.get_patch(START_X, START_Y, PATCH_WIDTH, PATCH_HEIGHT, SPACING)
    print(image.level_count, image.shapes, )
    print(patch.shape)
    try:
        os.mkdir(PATCH_DIRECTORY)
        print("Directory " , PATCH_DIRECTORY ,  " Created ") 
    except FileExistsError:
        # shutil.rmtree(PATCH_DIRECTORY)
        print("Directory " , PATCH_DIRECTORY ,  " already exists")
        # os.mkdir(PATCH_DIRECTORY)
    filename = PATCH_DIRECTORY + '/single_patch.jpg' 
    cv2.imwrite(filename, patch)

#generate random patches for a budget


if __name__ == '__main__': 
    if(IS_SINGLE): 
        generate_single_patch()
    else: 
        generate_all_patches()
    
    cv2.destroyAllWindows()