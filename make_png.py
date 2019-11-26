import torch
from torchvision.utils import save_image
import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import cv2
# for pretrained model
import torchvision.models as models

# load models and pretrained selector network
from models.nets import *
from models.selector import *
from data import *


# cv2.imread imwrite :: 64*(64*26)*3 format and BGR.
if __name__ == "__main__":
    data = cv2.imread('test_source.png', 1)
    # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    print (data.shape)
    data = torch.from_numpy(data).float() # 64*(64*26)*3

    position_list = alphabet_position('tlqkf')
    glyph_list = []
    for p in position_list:
        glyph_list.append(data[:,64*(p-1):64*p, :])
    output_glyph = torch.cat (glyph_list, dim=1)
    cv2.imwrite ('test_source_tlqkf.png', output_glyph.numpy())
