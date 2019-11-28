import torch
from torchvision.utils import save_image
import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import cv2
import matplotlib.pyplot as plt
import scipy.misc
# for pretrained model
import torchvision.models as models

# load models and pretrained selector network
from models.nets import *
from models.selector import *
from data import *


# cv2.imread imwrite :: 64*(64*26)*3 format and BGR.
if __name__ == "__main__":
    data = plt.imread('BLOOD.png')
    data = torch.from_numpy(data).float() # 64*(64*26)*3
    data = torch.unsqueeze(data.permute (2,0,1), 0) # 1*3*64*(64*26)

    position_list = alphabet_position('tlqkf')
    glyph_list = []
    for p in position_list:
        glyph_list.append(data[:,:,:,64*(p-1):64*p])
    output_glyph = torch.cat (glyph_list, dim=3)
    output_glyph = torch.squeeze (output_glyph)#.permute (1,2,0)
    save_image (output_glyph, 'BLOOD_tlqkf.png') # 3*64*(64*26)
    # scipy.misc.toimage(output_glyph.numpy()).save('scipy.png')
    # plt.imsave ('test_source_tlqkf_tlqkf.png', output_glyph.numpy())
