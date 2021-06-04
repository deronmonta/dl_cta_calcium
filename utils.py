import numpy as np
import SimpleITK as sitk
from math import ceil
import random

def pad_3d_array(arr, pad_values, patch_size=128, stride_size=96, phase='train', label_arr=None):
    """pad a 3d numpy array before patch extraction

    Arguments:
        arr {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    assert phase == 'test'
    
    def get_pad_size(stride_size, arr_shape, patch_size):
        pad = stride_size - ((arr_shape-patch_size) % stride_size)
        if pad == stride_size:
            pad = 0
        return pad
    
    print('Shape before padding: {}'.format(arr.shape))
    x_pad = get_pad_size(stride_size, arr.shape[0], patch_size)
    y_pad = get_pad_size(stride_size, arr.shape[1], patch_size)
    z_pad = get_pad_size(stride_size, arr.shape[2], patch_size)
    
    start_paddings = [ceil(x_pad/2), ceil(y_pad/2) ,ceil(z_pad/2)]
    end_paddings = [x_pad - start_paddings[0], y_pad - start_paddings[1], z_pad - start_paddings[2]]
    print('Amount of padding added on start side {}'.format(start_paddings))
    print('Amount of padding added on end side {}'.format(end_paddings))
    arr = np.pad(arr, ((start_paddings[0], end_paddings[0]), (start_paddings[1],  end_paddings[1]), (start_paddings[2], end_paddings[2])), 'constant', constant_values=((pad_values, pad_values), (pad_values, pad_values), (pad_values ,pad_values)))
    print('Shape after padding: {}'.format(arr.shape))

    return arr, start_paddings, end_paddings


def size_match(image_size, patch_size, stride_size):
    """Check if patch and stride sizes match with image size
    
    Args:
        image_size {tuple}: [description]
        patch_size {tuple}: [description]
        stride_size {tuple}: [description]
    
    Returns:
        bool: [description]
    """
    for i in range(len(image_size)):
        if not (image_size[i] - patch_size[i]) % stride_size[i] == 0:
            return False
    return True


def get_patch_num(image_size, patch_size, stride_size):
    """Get number of patches along each dimension
    
    Args:
        image_size {tuple}: [description]
        patch_size {tuple}: [description]
        stride_size {tuple}: [description]
    
    Returns:
        list: [description]
    """
    assert size_match(image_size, patch_size, stride_size)
    
    patch_nums = [(image_size[i] - patch_size[i]) // stride_size[i] + 1 for i in range(len(image_size))]
    return patch_nums
    

def patch_extract_3D(input,patch_shape=[128,128,128],stride_size=96):
    '''Extract patches from the whole image
    
    Args: 
        input {numpy.array}: Whole image
        patch_shape {tuple}: 
        xstep (int, optional): Stride along x. Defaults to 96.
        ystep (int, optional): Stride along y. Defaults to 96.
        zstep (int, optional): Stride along z. Defaults to 96.
        
    Returns: 
        numpy.array: Patches, [num of patches, x, y, z]
    '''
    assert size_match(input.shape, patch_shape, (stride_size, stride_size, stride_size))
    patch_nums = get_patch_num(input.shape, patch_shape, (stride_size, stride_size, stride_size))
    
    patches = np.zeros([np.prod(patch_nums), patch_shape[0], patch_shape[1], patch_shape[2]], input.dtype)
    
    i = 0
    for x in range(patch_nums[0]):
        xs = x * stride_size
        for y in range(patch_nums[1]):
            ys = y * stride_size
            for z in range(patch_nums[2]):
                zs = z * stride_size
                patches[i] = input[xs:xs+patch_shape[0], ys:ys+patch_shape[1], zs:zs+patch_shape[2]]
                i += 1
    return patches

def roi_extraction_train(og_img,label):

    label = label.astype(int)
    
    coord = np.where(label ==1)
    x_min, x_max, y_min, y_max, z_min, z_max = min(coord[0]), max(coord[0]), min(coord[1]), max(coord[1]),min(coord[2]), max(coord[2])

    roi_x_min = max((int(x_min)- (120 + random.randint(-8,8))), 0) # in case it goes to zero
    roi_x_max = int(x_max) + (120 + random.randint(-4,8))
    roi_y_min = max(y_min - (120 + random.randint(-4,8), 0))
    roi_y_max = y_max + (120 + random.randint(-4,8))
    roi_z_min = max(z_min - (120 + random.randint(-4,8), 0))
    roi_z_max = z_max + (120 + random.randint(-4,8))

    roi_artery_img = og_img[roi_x_min:roi_x_max, roi_y_min:roi_y_max, roi_z_min:roi_z_max]

    roi_cl_img = label[roi_x_min:roi_x_max, roi_y_min:roi_y_max, roi_z_min:roi_z_max]

    return roi_artery_img, roi_cl_img, roi_x_min, roi_x_max, roi_y_min, roi_y_max, roi_z_min, roi_z_max

def stuff_3D_patches_test(patches,out_shape,xstep=96,ystep=96,zstep=96):
    """Stuff the processed 3D patches back to original shape
    
    Args:
        patches {numpy.array}: Patches. [num of patches, x, y, z]
        out_shape {tuple}: Output image shape. [x, y, z]
        xstep (int, optional): Stride along x. Defaults to 96.
        ystep (int, optional): Stride along y. Defaults to 96.
        zstep (int, optional): Stride along z. Defaults to 96.
    
    Returns:
        [type]: [description]
    """
    
    out = np.zeros(out_shape, patches.dtype)
    denom = np.zeros(out_shape, patches.dtype)
    patch_shape = patches.shape[-3:]
    
    assert size_match(out_shape, patch_shape, (xstep, ystep, zstep))
    patch_nums = get_patch_num(out_shape, patch_shape, (xstep, ystep, zstep))
    
    i = 0
    for x in range(patch_nums[0]):
        xs = x * xstep
        for y in range(patch_nums[1]):
            ys = y * ystep
            for z in range(patch_nums[2]):
                zs = z * zstep
                out[xs:xs+patch_shape[0], ys:ys+patch_shape[1], zs:zs+patch_shape[2]] += patches[i]
                denom[xs:xs+patch_shape[0], ys:ys+patch_shape[1], zs:zs+patch_shape[2]] += 1
                i += 1
    return out/denom

def stuff_3D_patches_max(patches,out_shape,xstep=96,ystep=96,zstep=96):
    """Stuff the processed 3D patches back to original shape
    
    Args:
        patches {numpy.array}: Patches. [num of patches, x, y, z]
        out_shape {tuple}: Output image shape. [x, y, z]
        xstep (int, optional): Stride along x. Defaults to 96.
        ystep (int, optional): Stride along y. Defaults to 96.
        zstep (int, optional): Stride along z. Defaults to 96.
    
    Returns:
        [type]: [description]
    """
    
    out = np.zeros(out_shape, patches.dtype)
    denom = np.zeros(out_shape, patches.dtype)
    patch_shape = patches.shape[-3:]
    
    assert size_match(out_shape, patch_shape, (xstep, ystep, zstep))
    patch_nums = get_patch_num(out_shape, patch_shape, (xstep, ystep, zstep))
    
    i = 0
    for x in range(patch_nums[0]):
        xs = x * xstep
        for y in range(patch_nums[1]):
            ys = y * ystep
            for z in range(patch_nums[2]):
                zs = z * zstep
                out[xs:xs+patch_shape[0], ys:ys+patch_shape[1], zs:zs+patch_shape[2]] = np.maximum(out[xs:xs+patch_shape[0], ys:ys+patch_shape[1], zs:zs+patch_shape[2]], patches[i])
                # denom[xs:xs+patch_shape[0], ys:ys+patch_shape[1], zs:zs+patch_shape[2]] += 1
                i += 1
    return out


def stuff_3D_patches_weighted_avg(patches,weight_matrix,out_shape,xstep=96,ystep=96,zstep=96):
    """Stuff the processed 3D patches back to original shape
    
    Args:
        patches {numpy.array}: Patches. [num of patches, x, y, z]
        out_shape {tuple}: Output image shape. [x, y, z]
        xstep (int, optional): Stride along x. Defaults to 96.
        ystep (int, optional): Stride along y. Defaults to 96.
        zstep (int, optional): Stride along z. Defaults to 96.
    
    Returns:
        [type]: [description]
    """
    
    out = np.zeros(out_shape, patches.dtype)
    denom = np.zeros(out_shape, patches.dtype)
    patch_shape = patches.shape[-3:]
    
    assert size_match(out_shape, patch_shape, (xstep, ystep, zstep))
    patch_nums = get_patch_num(out_shape, patch_shape, (xstep, ystep, zstep))
    
    i = 0
    for x in range(patch_nums[0]):
        xs = x * xstep
        for y in range(patch_nums[1]):
            ys = y * ystep
            for z in range(patch_nums[2]):
                zs = z * zstep
                out[xs:xs+patch_shape[0], ys:ys+patch_shape[1], zs:zs+patch_shape[2]] += np.multiply(patches[i],weight_matrix)
                denom[xs:xs+patch_shape[0], ys:ys+patch_shape[1], zs:zs+patch_shape[2]] += weight_matrix
                i += 1
    return out/denom