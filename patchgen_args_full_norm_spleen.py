# generation of patches from a 'single' whole-slide svs file
# mask file is png file with binary annotations of tissue region vs. background
# coverage is a values between 0 and 1 to determine the threshold of accepting patches with 

import os
import openslide
from PIL import Image
import numpy as np
import cv2
import argparse
import datetime
Image.MAX_IMAGE_PIXELS = 1000000000000

parser = argparse.ArgumentParser()
parser.add_argument('--svs_file', type=str)
parser.add_argument('--mask_file', type=str)
parser.add_argument('--dest_dir', type=str)
parser.add_argument('--patch_size', type=int)
parser.add_argument('--overlap', type=int)
parser.add_argument('--coverage', type=float)

args = parser.parse_args()

svs_file = args.svs_file
mask_file = args.mask_file
dest_dir = args.dest_dir
patch_size = args.patch_size
overlap = args.overlap
cov = args.coverage

svs_name = os.path.basename( svs_file ).split(' ')[0]
    
slide = openslide.OpenSlide( svs_file )
    
if not os.path.exists(dest_dir): os.makedirs(dest_dir)
            
w, h = slide.dimensions
    
mask = np.array(Image.open(mask_file))
    
print('dimensions of', svs_name,'is',w,'by',h)
print(datetime.datetime.now())


def norm(patch):
    
    #target_stds = [39.12562752, 8.77984957, 10.4473662]
    #target_means = [123.10669392, 158.9313522, 101.15281356]
    
    target_stds = [40, 10, 10]
    target_means = [150, 160, 100]

    patch_np = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2LAB)
    patch_np = patch_np.astype(np.float32)
    source_means = [patch_np[..., i].mean() for i in range(3)]
    source_stds = [patch_np[..., i].std() for i in range(3)]
    for i in range(3):
        patch_np[:, :, i] = (patch_np[:, :, i] - source_means[i]) * target_stds[i] / source_stds[i] + target_means[i]
    patch_np=patch_np.clip(0,255)
    patch_normalized = cv2.cvtColor(patch_np.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    return(patch_normalized)

j = 1

for x0 in range(0, w-patch_size, patch_size-overlap):
    for y0 in range(0, h-patch_size, patch_size-overlap):
        mask_patch = mask[y0:y0+patch_size, x0:x0+patch_size]
        if np.count_nonzero(mask_patch)> cov*patch_size*patch_size:
            patch = slide.read_region((x0,y0), 1, (patch_size//4,patch_size//4))
            patch_normalized=norm(patch)
            cv2.imwrite( dest_dir + '/' + svs_name + '_%05d_x0_%02d_y0_%02d.jpg' %(j,x0, y0), patch_normalized)
            j += 1
            if (j%100==0):
                print(j)
        #if (j>20):
        #    break