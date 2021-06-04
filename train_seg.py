from unet3d.model import ResidualUNet3D_Segmentation
from unet3d.losses import DiceLoss, WeightedFocalLoss,WeightedBCE
from dataset import CAC_Segmentation_Dataset
import argparse
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms, utils
import numpy as np
import SimpleITK as sitk
from unet3d.dice_bce import DiceBCELoss

parser = argparse.ArgumentParser(description="")

# Training arguments
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--epochs", type=int, default=100,
                    help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")

# Model arguments
parser.add_argument(
    "--model_name", default="./model/CAC_segmentation_model.pkl", help="path to the model")

parser.add_argument(
    "--shuffle", action='store_true', help="path to the model")


cudnn.benchmark = True
args = parser.parse_args()
print(args)

train_set = CAC_Segmentation_Dataset()
data_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0,pin_memory=True)

model = ResidualUNet3D_Segmentation(in_channels=1, out_channels=1, final_sigmoid=True, f_maps=32, conv_layer_order='cge')
model.cuda()
criterion = WeightedBCE()

optimizer = optim.AdamW(
    model.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
    amsgrad=True,
    eps=1e-08,
    weight_decay=1e-6)

for epoch in range(0,args.epochs):
    for index, sample in tqdm(enumerate(data_loader)):
        img_patches = sample['img_patches']
        mask_patches = sample['mask_patches']

        print('Image shape {}'.format(img_patches.shape))

        img_patches = img_patches.float().cuda()
        mask_patches = mask_patches.float().cuda()

        running_loss = 0.0
        print('Number of patches {}'.format(img_patches.size(1)))

        for patch_num in tqdm(range(img_patches.size(1))): # second dimension is the number of patches, first dimension is batchsize
            
            img_patch = img_patches[:,patch_num, :, :, :] # second 
            label_patch = mask_patches[:,patch_num, :, :, :]
            label_patch = torch.unsqueeze(label_patch, 0)


            input_patch = torch.unsqueeze(input_patch,0)
            output = model(input_patch)

            loss = criterion(output, label_patch)
            running_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Average patient Loss: {:.6f}'.format(
                running_loss/img_patches.size(1)))

        # save a checkpoint model every x data 
        torch.save(model, args.model_name)

        if index % 10 == 0:
            model_name = args.model_name.replace('.pkl', '_{}.pkl'.format(index))
            torch.save(model, model_name)
