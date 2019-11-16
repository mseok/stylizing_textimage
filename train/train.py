# for pretrained model
import torchvision.models as models

# load models and pretrained selector network
from ..models.nets import *
from ..models.selector import *

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Data loader


# pretrained model for content selector
pt = models.vgg16(pretrained=True).features.eval()
for source in train_dataset:
    model, content_losses = transfer_model(pt, source)

for glyph in glyph_dataset:

model, content_losses = transfer_model(pt, style_img)
model(content_img)
content_score = 0
for i, sl in enumerate(content_losses):
    content_score += 100 * sl.loss
content_score
