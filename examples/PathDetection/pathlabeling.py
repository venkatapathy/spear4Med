from wholeslidedata.image.wholeslideimage import WholeSlideImage
from matplotlib import pyplot as plt

# open a WSI
with WholeSlideImage('data/036708CV__20220818_182724.tiff', backend='openslide') as wsi:
    print('\n\nWSI properties\n---------------\n')
    # print some properties
    print(f'available spacing in {wsi.path}:\n{wsi.spacings}\n')
    print(f'shapes in {wsi.path}:\n{wsi.shapes}\n')
    print(f'downsampling ratios in {wsi.path}:\n{wsi.downsamplings}\n')
    print(f'closest real spacing from rounded spacing: 0.5 = {wsi.get_real_spacing(0.5)}\n')
    print(f'size given spacing: shape at spacing 0.5 = {wsi.shapes[wsi.get_level_from_spacing(0.5)]}\n')

    # extract a patch with center coordinates xy at spacing 0.5
    x,y = 19000, 12000
    width, height = 1024, 1024
    spacing = 0.5
    patch = wsi.get_patch(x, y, width, height, spacing)
    plt.imshow(patch)
    plt.show()