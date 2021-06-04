import torch
import numpy as np
import SimpleITK as sitk
import pandas as pd
from glob import glob
from tqdm import tqdm # For progress bars
import os
from math import ceil
import cc3d
import random 
from scipy.ndimage import zoom, binary_dilation
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils
from torch.nn import functional as F
import argparse
import numpy as np
from tqdm import tqdm
import os
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_erosion
from scipy.ndimage import zoom
from utils import img_loading_preprocess,roi_extraction,resizeImage_SimpleITK
from patch_utils import *
import time

import itertools


def bbox_ND(img):
    """get bounding box of ND image
    returns (min_1, max_1, min_2, max_2, ...)
    https://stackoverflow.com/a/31402351"""
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    return tuple(out)

def sortCompBySize(npArr, min_size):
    """
    input: binary suspect mask (np.array)
    output: instance-ID suspect mask (np.array), components with < 1000 voxels removed
    """
    # sort components from largest to smallest. remove comps < 1000 voxels 
    im = sitk.GetImageFromArray(npArr.astype('uint8'))
    t_start = time.time()
    ccIm = sitk.ConnectedComponent(im)
    print("cc time", time.time()-t_start)
    
    t_start = time.time()
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkUInt16)
    ccIm = castImageFilter.Execute(ccIm)
    outputIm = sitk.RelabelComponent(ccIm, minimumObjectSize=min_size, sortByObjectSize=True)
    print("sort and remove comp time", time.time()-t_start)
    outputArr = sitk.GetArrayFromImage(outputIm)
    return outputArr

parser = argparse.ArgumentParser(description="")

parser.add_argument("--itkimg_dir", default="/data/home/haoyuy/coronary_artery_calcium/sevre_case/CTA_iso/case0034_iso.nii.gz")
parser.add_argument("--artery_mask_dir", default='/data/home/haoyuy/coronary_artery_calcium/sevre_case/CTA_iso_seg/case0034_artery_iso.nii.gz')

parser.add_argument("--pred_filename",default="/data/home/haoyuy/coronary_artery_calcium/case0034_cal_pred.nii.gz")
# Model arguments
parser.add_argument(
    "--CAC_model", default="/data/home/haoyuy/coronary_artery_calcium/model/CAC_segmentation_model.pkl", help="path to the model")

cudnn.benchmark = True
args = parser.parse_args()
'''
model 
'''

print(args.itkimg_dir)
CAC_model = torch.load(args.CAC_model)
CAC_model.cuda()
CAC_model.eval()
for param in CAC_model.parameters():
    param.requires_grad = False
img_itk = sitk.ReadImage(args.itkimg_dir)
img_arr = sitk.GetArrayFromImage(img_itk)

artery_itk = sitk.ReadImage(args.artery_mask_dir)
artery_arr_og = sitk.GetArrayFromImage(artery_itk)




artery_arr_og[artery_arr_og==1] = 0
artery_arr_og[artery_arr_og==2] = 1
artery_arr_og[artery_arr_og==3] = 1
artery_arr = np.copy(artery_arr_og)


artery_arr = artery_arr.astype(np.int8)
artery_arr = binary_dilation(artery_arr, iterations=7)
artery_arr = artery_arr.astype(np.int8)
artery_arr[artery_arr==1] = 200
artery_arr[artery_arr==1] = 200

img_arr, start_paddings, end_paddings = pad_3d_array(img_arr, pad_values=200, patch_size=128, stride_size=96,phase='test')
artery_arr, start_paddings, end_paddings= pad_3d_array(artery_arr, pad_values=100, patch_size=128, stride_size=96,phase='test')

padding_added_shape = img_arr.shape

img_patches = patch_extract_3D(img_arr,patch_shape=[128,128,128],stride_size=96)
artery_patches = patch_extract_3D(artery_arr,patch_shape=[128,128,128],stride_size=96)

img_patches = np.clip(img_patches, 200,1000)

img_patches = torch.from_numpy(img_patches).float().cuda()
artery_patches = torch.from_numpy(artery_patches).float().cuda()


whole_pred = torch.zeros(img_patches.shape[0], img_patches.shape[1], img_patches.shape[2], img_patches.shape[3])

for patch_num in tqdm(range(img_patches.size(0))): # second dimension is the number of patches, first dimension is batchsize

    img_patch = img_patches[patch_num, :, :, :] # second 
    artery_patch = artery_patches[patch_num, :, :, :] # second 

    input = torch.stack((img_patch, artery_patch),dim=0)
    input = torch.unsqueeze(input, dim=0)
    output = CAC_model(input)
    whole_pred[patch_num, :, :, :] = output



whole_pred =  whole_pred.data.cpu().numpy()


pred_out = stuff_3D_patches_max(whole_pred, out_shape=padding_added_shape, xstep=96,ystep=96,zstep=96)

pred_out = pred_out[start_paddings[0]:pred_out.shape[0]-end_paddings[0], start_paddings[1]:pred_out.shape[1]-end_paddings[1], start_paddings[2]:pred_out.shape[2]-end_paddings[2]]

artery_arr = artery_arr[start_paddings[0]:artery_arr.shape[0]-end_paddings[0], start_paddings[1]:artery_arr.shape[1]-end_paddings[1], start_paddings[2]:artery_arr.shape[2]-end_paddings[2]]



# print(np.ptp(artery_arr))
# print(np.ptp(artery_arr))
# artery_arr = artery_arr/200
print(np.ptp(artery_arr_og))
artery_arr_large = binary_dilation(artery_arr_og, iterations=5)
pred_out = np.multiply(pred_out,artery_arr_large)


# #invert the artery
# artery_arr_small = np.copy(artery_arr)
# artery_arr_small = artery_arr_small.astype(np.uint8)

# artery_arr_small[artery_arr_small==1]= 10
# artery_arr_small[artery_arr_small ==0] = 1
# artery_arr_small[artery_arr_small ==10] = 0

# artery_arr_small_itk = sitk.GetImageFromArray(artery_arr_small)
# sitk.WriteImage(artery_arr_small_itk, 'invert_mask.nii.gz')
#pred_out = np.multiply(pred_out, artery_arr_small)


print(np.ptp(pred_out))

pred_out[pred_out >  0.5] = 1
pred_out[pred_out < 0.5] = 0

pred_out = pred_out.astype(np.int8)



print('Artery Stuffed shape {}'.format(pred_out.shape))

###########################################
# Post process Artery
##########################################

# remove components smaller than 2000 voxels



pred_out_CC = sortCompBySize(pred_out, min_size=5)
pred_out_CC[pred_out_CC != 0] =1

# label left/right
# pred_out.fill(0)

# mean_x_coords = np.mean(np.where(pred_out_CC>0)[2]) # mean of all y coords in all cc
# print('Mid point {}'.format(mean_x_coords))
# for index in range(1, np.amax(pred_out_CC)+1): 
#     if np.mean(np.where(pred_out_CC == index)[2]) > mean_x_coords: 
#         pred_out[pred_out_CC==index] = 2
#         print('Index {} labeled as lca'.format(index))
#     else: 
#         pred_out[pred_out_CC==index] = 3
#         print('Index {} labeled as rca'.format(index))


# # merge artery to aorta mask
# og_shape_pred = og_shape_pred_aorta # og_shape_pred is simply a reference to og_shape_pred_aorta, to avoid allocating a new big image

# artery_roi_slice = slice(roi_x_min,roi_x_max), slice(roi_y_min,roi_y_max), slice(roi_z_min,roi_z_max)
# np.maximum(og_shape_pred[artery_roi_slice], pred_out, out=og_shape_pred[artery_roi_slice]) # put new aorta back


# getting rid of thin layer around ostia
# First get an overlapping version
# aorta_box = bbox_ND(og_shape_pred_aorta)
# margin = 3
# aorta_box = (   max(0, aorta_box[0] - margin), min(og_shape_pred_aorta.shape[0] - 1, aorta_box[1] + margin), \
#                 max(0, aorta_box[2] - margin), min(og_shape_pred_aorta.shape[1] - 1, aorta_box[3] + margin), \
#                 max(0, aorta_box[4] - margin), min(og_shape_pred_aorta.shape[2] - 1, aorta_box[5] + margin))

# aorta_box_slice = slice(aorta_box[0], aorta_box[1] + 1), slice(aorta_box[2],aorta_box[3]+1), slice(aorta_box[4],aorta_box[5]+1)

# aorta_roi = (og_shape_pred == 1)[aorta_box_slice]
# aorta_roi = binary_erosion(aorta_roi, iterations=3)
# aorta_roi = binary_dilation(aorta_roi, iterations=3)

# # put back aorta
# og_shape_pred[og_shape_pred==1] = 0 # erase old aorta
# np.maximum(og_shape_pred[aorta_box_slice], aorta_roi, out=og_shape_pred[aorta_box_slice]) # put new aorta back

# print('Aorta length {}'.format(og_shape_pred.shape[0] - min(np.where(og_shape_pred ==1)[0])))
# aorta_len = og_shape_pred.shape[0] - min(np.where(og_shape_pred ==1)[0])

# if aorta_len > 140:
#     print('Cropping aorta') 
#     target_max = (min(np.where(og_shape_pred ==1)[0]) + 140)
#     crop_len = og_shape_pred.shape[0] - target_max
#     print('Crop len: {}'.format(crop_len))
#     og_shape_pred[-crop_len:-1, :, :] = 0 
# og_shape_pred[-1, :, :] = 0


artery_coords = np.where(artery_arr_og==1)
pred_out_CC[artery_coords] = 2
pred_out_CC = pred_out_CC.astype(np.uint8)
itk_pred = sitk.GetImageFromArray(pred_out_CC)
itk_pred.CopyInformation(artery_itk)
sitk.WriteImage(itk_pred, args.pred_filename)
