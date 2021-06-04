from unet3d.model import ResidualNet3D_Regression
from unet3d.losses import DiceLoss, WeightedFocalLoss,WeightedBCE
from dataset import CAC_Regression_Dataset
import argparse
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms, utils
import os
import numpy as np
import SimpleITK as sitk
from utils import patch_extract_3D
from scipy.ndimage import zoom, binary_dilation
import random
from unet3d.dice_bce import DiceBCELoss

parser = argparse.ArgumentParser(description="")

# Training arguments
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--epochs", type=int, default=100,
                    help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")

# Model arguments
parser.add_argument(
    "--model_name", default="./model/CAC_regression_model_ckpt.pkl", help="path to the model")

parser.add_argument(
    "--shuffle", action='store_true', help="path to the model")


cudnn.benchmark = True
args = parser.parse_args()
print(args)

train_set = CAC_Regression_Dataset()
data_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0,pin_memory=True)

model = ResidualNet3D_Regression(in_channels=3, out_channels=1, final_sigmoid=False, f_maps=16, conv_layer_order='cge')
model = torch.load(args.model_name)
model.cuda()


criterion = nn.MSELoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
    amsgrad=True,
    eps=1e-08,
    weight_decay=1e-6)

for epoch in range(0,args.epochs):
    for index, sample in tqdm(enumerate(data_loader)):
        channel_1_img_patches = sample['channel_1_img_patches']
        channel_2_img_patches = sample['channel_2_img_patches']
        cal_patches = sample['cal_patches']
        cal_seg_patches = sample['cal_seg_patches']

        channel_1_img_patches = channel_1_img_patches.float().cuda()
        channel_2_img_patches = channel_2_img_patches.float().cuda()
        cal_patches = cal_patches.float().cuda()
        cal_seg_patches = cal_seg_patches.float().cuda()

        running_loss = 0.0
        print('Number of patches {}'.format(channel_1_img_patches.size(1)))

        for patch_num in tqdm(range(channel_1_img_patches.size(1))): # second dimension is the number of patches, first dimension is batchsize
            
            channel_1_patch = channel_1_img_patches[:,patch_num, :, :, :] # second 
            channel_2_patch = channel_2_img_patches[:,patch_num, :, :, :] # second 
            cal_patch = cal_patches[:,patch_num, :, :, :]
            cal_seg_patch = cal_seg_patches[:,patch_num, :, :, :]
            
            input_patch = torch.stack((channel_1_patch, channel_2_patch, cal_seg_patch), dim=1)
            print(input_patch.shape)
            
            CAC_score = torch.sum(cal_patch)
            print('CAC in this patch: {}'.format(CAC_score))

            output = model(input_patch)
            CAC_score = CAC_score.cuda()
            loss = criterion(output, CAC_score)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Average patient Loss: {:.6f}'.format(
                running_loss/channel_1_img_patches.size(1)))

        # save a checkpoint model every x data 
        torch.save(model, args.model_name)

        if index % 10 == 0:
            model_name = args.model_name.replace('.pkl', '_{}.pkl'.format(index))
            torch.save(model, model_name)
