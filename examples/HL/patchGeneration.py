import argparse

parser = argparse.ArgumentParser(description='Patch Generate')
openslide_path = parser.add_argument('--openslide_path', metavar='OpenSlide', type=str, 
                    help='OpenSlide bin path')
function = parser.add_argument('--function', metavar='F', type=str, 
                    help='function to be called')
tiffPath = parser.add_argument('--tiffPath', metavar='tif', type=str, 
                    help='tiff image path')
parser.add_argument('--patch_directory', metavar='Output', default='patches', type=str, 
                    help='output patch directory')
parser.add_argument('--spacing', metavar='S', default=0.5, type=float, 
                    help='image spacing')
parser.add_argument('--patch_height', metavar='H', default=256, type=int, 
                    help='height of the image patch')
parser.add_argument('--patch_width', metavar='W', default=256, type=int, 
                    help='width of the image patch')
startX = parser.add_argument('--startX', metavar='X', type=float, 
                    help='centerX of the image patch')
startY = parser.add_argument('--startY', metavar='Y', type=float, 
                    help='centerY of the image patch')

args = parser.parse_args()


'''Show Errors if any of these are not specified'''
if args.tiffPath is None:
    raise argparse.ArgumentError(tiffPath,
                                 'tiff Path was not specified')

if args.function is None:
    raise argparse.ArgumentError(function,
                                 'Function to be invoked must be specifid, should be single, random or all')


import os
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(args.openslide_path):
        import openslide
else:
    import openslide


import time
import random
import cv2
import shutil
from wholeslidedata.image.wholeslideimage import WholeSlideImage
from wholeslidedata.iterators import create_patch_iterator


'''Generate All Patches'''
def generate_all_patches(path, 
    spacing, 
    patch_directory, 
    patch_height, 
    patch_width
):
    image = WholeSlideImage(path)
    dimensions = image.get_shape_from_spacing(spacing)

    print(dimensions[0], dimensions[1])
    try:
        os.mkdir(patch_directory)
        print("Directory " , patch_directory ,  " Created ") 
    except FileExistsError:
        print("Directory " , patch_directory ,  " already exists")

    t1 = time.time()
    count = 0
    for startY in range(0, dimensions[1], patch_height):
        for startX in range(0, dimensions[0], patch_width):
            count+=1
            patch = image.get_patch(startX, startY, patch_width, patch_height, spacing)
            cv2.imwrite(patch_directory + f'/patch_{count}.jpg', patch)
    print(f'Total Time: {time.time() - t1}')


'''Generate Single Patch'''
def generate_single_patch(path, 
    spacing, 
    patch_directory, 
    patch_height, 
    patch_width,
    start_x,
    start_y
):
    if args.startX is None or args.startY is None:
        raise argparse.ArgumentError(openslide_path,
                                 'Please specifiy X and Y co-ordinates')
    
    image = WholeSlideImage(path)
    patch = image.get_patch(start_x, start_y, patch_width, patch_height, spacing)

    try:
        os.mkdir(patch_directory)
        print("Directory " , patch_directory ,  " Created ") 
    except FileExistsError:
        print("Directory " , patch_directory ,  " already exists")

    filename = patch_directory + '/single_patch.jpg' 
    cv2.imwrite(filename, patch)


'''Generate Random Patch'''
def generate_random_patch(path, 
    spacing, 
    patch_directory, 
    patch_height, 
    patch_width,
):
    image = WholeSlideImage(path)
    dimensions = image.get_shape_from_spacing(spacing)

    start_x = random.randint(patch_width, dimensions[0] - patch_width)
    start_y = random.randint(patch_height, dimensions[1] - patch_height)

    patch = image.get_patch(start_x, start_y, patch_width, patch_height, spacing)

    try:
        os.mkdir(patch_directory)
        print("Directory " , patch_directory ,  " Created ") 
    except FileExistsError:
        print("Directory " , patch_directory ,  " already exists")

    filename = patch_directory + '/random_patch.jpg' 
    cv2.imwrite(filename, patch)


if(args.function == 'single'):
    generate_single_patch(args.tiffPath, args.spacing, args.patch_directory, 
                args.patch_height, args.patch_width, args.startX, args.startY)

elif(args.function == 'random'):
    generate_random_patch(args.tiffPath, args.spacing, args.patch_directory, 
                args.patch_height, args.patch_width)

elif(args.function == 'all'):
    generate_all_patches(args.tiffPath, args.spacing, args.patch_directory, 
                args.patch_height, args.patch_width)

else:
    print('Please enter a valid function name: single or random or all')