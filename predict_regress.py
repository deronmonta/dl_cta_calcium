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

# Model arguments
parser.add_argument(
    "--CAC_model", default="/data/home/haoyuy/coronary_artery_calcium/model/CAC_regression_model.pkl", help="path to the model")

cudnn.benchmark = True
args = parser.parse_args()
'''
model 
'''

CAC_model = torch.load(args.CAC_model)
CAC_model.cuda()
CAC_model.eval()
for param in CAC_model.parameters():
    param.requires_grad = False

pred_lis = []

for caseID in caseID_lis:

    img_itk = sitk.ReadImage(args.itkimg_dir)
    img_arr = sitk.GetArrayFromImage(img_itk)

    cal_seg_itk = sitk.ReadImage(args.cal_seg_dir)
    cal_seg_arr = sitk.GetArrayFromImage(cal_seg_itk)

    img_arr[img_arr > 1000] = 1000
    img_arr[img_arr < 0] = 0
    cal_seg_arr = cal_seg_arr*100


    img_arr, start_paddings, end_paddings = pad_3d_array(img_arr, pad_values=0, patch_size=128, stride_size=96,phase='test')
    cal_seg_arr, _, _ = pad_3d_array(cal_seg_arr, pad_values=0, patch_size=128, stride_size=96,phase='test')

    padding_added_shape = img_arr.shape

    img_patches = patch_extract_3D(img_arr,patch_shape=[128,128,128],stride_size=96)
    cal_seg_patches = patch_extract_3D(cal_seg_arr,patch_shape=[128,128,128],stride_size=96)

    img_patches = torch.from_numpy(img_patches).float().cuda()
    cal_seg_patches = torch.from_numpy(cal_seg_patches).float().cuda()

    whole_pred = torch.zeros(img_patches.shape[0], img_patches.shape[1], img_patches.shape[2], img_patches.shape[3])

    for patch_num in tqdm(range(img_patches.size(0))): # second dimension is the number of patches, first dimension is batchsize
        if torch.sum(cal_seg_patch) == 0:

            img_patch = img_patches[patch_num, :, :, :] 
            cal_seg_patch = cal_seg_patches[patch_num, :, :, :] 
            
            input_patch = torch.stack((img_patch, cal_seg_patch), dim=0)
            input_patch = torch.unsqueeze(input_patch, dim=0)
            output = CAC_model(input_patch)
            
            whole_pred[patch_num, :, :, :] = output
        else:
            print('No calcifications')
            output = torch.zeros((128,128,128)).float().cuda()
            whole_pred[patch_num, :, :, :] = output


    whole_pred =  whole_pred.data.cpu().numpy()
    print(np.ptp(whole_pred))

    pred_out = stuff_3D_patches_test(whole_pred, out_shape=padding_added_shape, xstep=96,ystep=96,zstep=96)
    pred_out = pred_out[start_paddings[0]:pred_out.shape[0]-end_paddings[0], start_paddings[1]:pred_out.shape[1]-end_paddings[1], start_paddings[2]:pred_out.shape[2]-end_paddings[2]]

    pred_sum = np.sum(pred_out)
    pred_lis.append(pred_sum)

    pred_out = pred_out.astype(np.float32)
    itk_pred = sitk.GetImageFromArray(pred_out)
    itk_pred.CopyInformation(img_itk)
    sitk.WriteImage(itk_pred, args.pred_filename)

df = pd.DataFrame({'caseID':caseID_lis, 'PredictionScore':pred_lis})
df.to_csv('CAC_score_prediction.csv',index=False)