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


use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=5, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=3, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--num_superpixels', metavar='K', default=1000, type=int, 
                    help='number of superpixels')
parser.add_argument('--compactness', metavar='C', default=1, type=float, 
                    help='compactness of superpixels')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name')
parser.add_argument('--patchno', metavar='PATCHNO',default=1958, type=int,help='Which patch number to choose from lysto')
args = parser.parse_args()

ds = h5py.File('/home/venkat/Projects/workbook/spear4Med/examples/PathDetection/data/training.h5', 'r')
cv2.imwrite('sample.jpg',cv2.cvtColor(ds['x'][args.patchno], cv2.COLOR_RGB2BGR))


print('Organ',ds['organ'][args.patchno])

ihc_hed = rgb2hed(ds['x'][args.patchno])
null = np.zeros_like(ds['x'][args.patchno][:, :, 0])
ihc_d = hed2rgb(np.stack((null, null, ds['x'][args.patchno][:, :, 2]), axis=-1))
ihc_e = hed2rgb(np.stack((null, ds['x'][args.patchno][:, :, 1],null), axis=-1))
ihc_h = hed2rgb(np.stack((ds['x'][args.patchno][:, :, 0],null,null), axis=-1))

ihc_d_gray=rgb2gray(ihc_e)
thresh = threshold_otsu(ihc_d_gray)
mask = ihc_d_gray < thresh
mask=morphology.remove_small_holes(mask,10)
#mask = morphology.remove_small_objects(
#        mask, 30)
# Rescale hematoxylin and DAB channels and give them a fluorescence look
h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1),
                      in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1),
                      in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))

zdh = np.dstack((null, d, null))

# slic
labels = segmentation.slic(ds['x'][args.patchno], mask=mask, n_segments=args.num_superpixels)
#labels = segmentation.slic(ds['x'][args.patchno], compactness=args.compactness, n_segments=args.num_superpixels)

data = torch.from_numpy( np.array([ihc_e.transpose((2,0,1)).astype('float32')/255.]) )

# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, args.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(args.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(args.nConv-1):
            self.conv2.append( nn.Conv2d(args.nChannel, args.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(args.nChannel) )
        self.conv3 = nn.Conv2d(args.nChannel, args.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(args.nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(args.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


if use_cuda:
    data = data.cuda()
data = Variable(data)


labels = labels.reshape(ds['x'][args.patchno].shape[0]*ds['x'][args.patchno].shape[1])
u_labels = np.unique(labels)
l_inds = []
for i in range(len(u_labels)):
    l_inds.append( np.where( labels == u_labels[ i ] )[ 0 ] )

# train
model = MyNet( data.size(1) )
if use_cuda:
    model.cuda()
model.train()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
label_colours = np.random.randint(255,size=(100,3))
for batch_idx in range(args.maxIter):
    # forwarding
    optimizer.zero_grad()
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    nLabels = len(np.unique(im_target))
    if args.visualize:
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( ds['x'][args.patchno].shape ).astype( np.uint8 )
        cv2.imshow( "output", im_target_rgb )
        cv2.waitKey(10)

    # superpixel refinement
    # TODO: use Torch Variable instead of numpy for faster calculation
    for i in range(len(l_inds)):
        labels_per_sp = im_target[ l_inds[ i ] ]
        u_labels_per_sp = np.unique( labels_per_sp )
        hist = np.zeros( len(u_labels_per_sp) )
        for j in range(len(hist)):
            hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
        im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
    target = torch.from_numpy( im_target )
    if use_cuda:
        target = target.cuda()
    target = Variable( target )
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    #print (batch_idx, '/', args.maxIter, ':', nLabels, loss.data[0])
    #print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())

    if nLabels <= args.minLabels:
        print ("nLabels", nLabels, "reached minLabels", args.minLabels, ".")
        break

# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( ds['x'][args.patchno].shape ).astype( np.uint8 )
cv2.imwrite( "output.png", im_target_rgb )
