# Hodgkin's Lymphoma

## Directory Structure
```
.
+-- archive
|   +-- old_files_backup
+-- images
|   +-- sample_patches
+-- notebooks
|   +-- latest_code_notebooks
+-- rsc_process.py
+-- eosinophil_process.py
+-- patchGeneration.py
```

## Patch Generation from Whole Slide Images

To generate patches from Whole Slide Images(WSI) we use the script patchGeneration.py. It contains three types of functions:
1. Single Patch: Specify the following parameters - openslide_patch(for Windows), tiffPath, startX, startY, function = 'single'. You can also pass other arguments as well as per your needs.

2. Random Patches: Specify the following parameters - openslide_patch(for Windows), tiffPath, function = 'random'. You can also pass other arguments as well as per your needs.

3. All Patches: Specify the following parameters - openslide_patch(for Windows), tiffPath, function = 'all'. You can also pass other arguments as well as per your needs. Download Openslide binaries from https://openslide.org/download/

4. Arguments:
    * openslide_path - OpenSlide bin path
    * function - 3 types of function(single, all, random)
    * tiffPath - tiff image path
    * patch_directory - output directory name to store the patches
    * spacing - image spacing, default  = 0.5
    * patch_height - height of the patch, default = 256
    * patch_width - width of the patch, default = 256
    * startX - x-coordinate of the center of the patch
    * startY - y-coordinate of the center of the patch

Example:
~~~~
python patchGeneration.py --openslide_path 'path\to\openslide\bin' --function 'random' --tiffPath 'wsi_name.tiff' --patch_directory 'path\to\patch_directory' --patch_height 256 --patch_width 256 --startX 35000 --startY 26000
~~~~

## Running Pre-Processing Scripts
The project currently contains two pre-processing scripts, one each for detecting Reed Sternberg Cells (RSCs) and Eosionphils using image processing techniques. Each of these files may be given two inputs:

1. Image File Path: Each script requires the complete path of the input image of a patch that needs to be processed. If this argument is not provided, the script runs on a default image, present in the images directory.

2. Output Directory Path: Each script generates three types of outputs that are stored in the output directory. If the provided directory does not already exist, the directory will be created by the script. If the directory already exists then files present within it may be overwritten. If this argument is not provided, the script generates an output directory called 'output' in the current working directory. The three types of output are:
    * Original Image: The original iamge is stored in the output directory.
    * Cell Crops: The Cells(RSCs & Eosinophils) detected by the first step of the processing pipeline is stored in a directory called 'cell' within the output directory.
    * Cell with Nucleoli Crops: The nucleoli detected by the second step of the processing pipeline are annotated on the cell crop. These images are stored in a directory called 'nucleolus' within the output directory.
    * Output Directory Structure:
        ```
        .
        +-- cell
        +-- nucleolus
        +-- original_patch.jpg
        ```

The commands for running the two scripts are as given below:

1. rsc_process.py
~~~~
python rsc_process.py --imagePath path\to\input\image.jpg --outputDir path\to\output_directory
~~~~

2. eosinophil_process.py
~~~~
python eosinophil_process.py --imagePath path\to\input\image.jpg --outputDir path\to\output_directory
~~~~