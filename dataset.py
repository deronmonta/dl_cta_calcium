import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
#from utils import , patch_extract_3D_test, pad_3d_array
import os
import SimpleITK as sitk
from scipy import ndimage, misc
import torchvision.transforms.functional as F
from scipy.ndimage import zoom
import random
from patch_utils import patch_extract_3D, pad_3d_array
from utils import roi_extraction_train
from scipy.ndimage import zoom, binary_dilation

class CAC_Segmentation_Dataset(Dataset):
    '''
    
    '''
    def __init__(
        self):

        text_file = open("train_ID_lis.txt", "r")
        self.id_lis = text_file.read().split('\n')
        print('Number of training files {}'.format(len(self.id_lis)))
    
    def __getitem__(self, index):

        caseID = self.id_lis[index]
        print(caseID)
        img_dir = '/data/home/haoyuy/coronary_artery_calcium/sevre_case/CTA_iso/{}_iso.nii.gz'.format(caseID)
        img_itk = sitk.ReadImage(img_dir)
        img_arr = sitk.GetArrayFromImage(img_itk)

        label_itk = sitk.ReadImage('/data/home/haoyuy/coronary_artery_calcium/sevre_case/CTA_iso_seg/{}_cal_seg.nii.gz'.format(caseID)) 
        cal_arr = sitk.GetArrayFromImage(label_itk)      
        print(np.ptp(cal_arr))
        stride_size = 96 + random.randint(-8,8)
        print('Stride size {}'.format(stride_size))

        channel_1_img = np.copy(img_arr)
        channel_1_img = np.clip(channel_1_img,-100,1000)
        channel_1_img = channel_1_img / 1100
        
        channel_2_img = np.copy(img_arr)
        channel_2_img = np.clip(channel_2_img,400,1000)
        channel_2_img = channel_2_img / 600

        channel_1_img, _, _ = pad_3d_array(channel_1_img, pad_values=0, patch_size=128, stride_size=stride_size,phase='test')
        channel_2_img, _, _ = pad_3d_array(channel_2_img, pad_values=0, patch_size=128, stride_size=stride_size,phase='test')
        mask_arr, _, _ = pad_3d_array(cal_arr, pad_values=0, patch_size=128, stride_size=stride_size,phase='test')

        channel_1_img_patches = patch_extract_3D(channel_1_img,patch_shape=[128,128,128],stride_size=stride_size)
        channel_2_img_patches = patch_extract_3D(channel_2_img,patch_shape=[128,128,128],stride_size=stride_size)
        mask_patches = patch_extract_3D(mask_arr,patch_shape=[128,128,128],stride_size=stride_size)
        
        print(np.ptp(mask_patches))
      

        sample = {'channel_1_img_patches': channel_1_img_patches, 'channel_2_img_patches':channel_2_img_patches, 'mask_patches':mask_patches, 'caseID':caseID}

        return sample
    
    def __len__(self):
        return len(self.id_lis)


class CAC_Regression_Dataset(Dataset):
    '''
    
    '''
    def __init__(
        self):

        text_file = open("train_ID_lis.txt", "r")
        self.id_lis = text_file.read().split('\n')
        print('Number of training files {}'.format(len(self.id_lis)))
    
    def __getitem__(self, index):

        caseID = self.id_lis[index]
        print(caseID)

        cal_seg_itk = sitk.ReadImage('/data/home/haoyuy/coronary_artery_calcium/sevre_case/CTA_iso_seg/{}_cal_seg.nii.gz'.format(caseID))
        cal_seg_arr = sitk.GetArrayFromImage(cal_seg_itk)

        cal_itk = sitk.ReadImage('/data/home/haoyuy/coronary_artery_calcium/sevre_case/CTA_iso_seg/{}_cal_map.nii.gz'.format(caseID))
        cal_arr = sitk.GetArrayFromImage(cal_itk)

        img_itk = sitk.ReadImage('/data/home/haoyuy/coronary_artery_calcium/sevre_case/CTA_iso/{}_iso.nii.gz'.format(caseID))
        img_arr = sitk.GetArrayFromImage(img_itk)

        # print(img_arr)
        stride_size = 128
        print('Stride size {}'.format(stride_size))

        channel_1_img = np.copy(img_arr)
        channel_1_img = np.clip(channel_1_img,-100,1000)
        channel_1_img = channel_1_img / 1100
        
        channel_2_img = np.copy(img_arr)
        channel_2_img = np.clip(channel_2_img,400,1000)
        channel_2_img = channel_2_img / 600

        cal_seg_arr = cal_seg_arr*100
        print(np.ptp(cal_arr))


        channel_1_img, _, _ = pad_3d_array(channel_1_img, pad_values=0, patch_size=128, stride_size=stride_size,phase='test')
        channel_2_img, _, _ = pad_3d_array(channel_2_img, pad_values=0, patch_size=128, stride_size=stride_size,phase='test')
        cal_arr, _, _ = pad_3d_array(cal_arr, pad_values=0, patch_size=128, stride_size=stride_size,phase='test')
        cal_seg_arr, _, _ = pad_3d_array(cal_seg_arr, pad_values=0, patch_size=128, stride_size=stride_size,phase='test')

        channel_1_img_patches = patch_extract_3D(channel_1_img,patch_shape=[128,128,128],stride_size=stride_size)
        channel_2_img_patches = patch_extract_3D(channel_2_img,patch_shape=[128,128,128],stride_size=stride_size)
        cal_patches = patch_extract_3D(cal_arr,patch_shape=[128,128,128],stride_size=stride_size)
        cal_seg_patches = patch_extract_3D(cal_seg_arr,patch_shape=[128,128,128],stride_size=stride_size)

        channel_1_img_patches = channel_1_img_patches.astype(np.float64)
        channel_2_img_patches = channel_2_img_patches.astype(np.float64)

        sample = {'channel_1_img_patches': channel_1_img_patches, 'channel_2_img_patches': channel_2_img_patches,'cal_patches':cal_patches, 'cal_seg_patches':cal_seg_patches}

        return sample
    
    def __len__(self):
        return len(self.id_lis)

