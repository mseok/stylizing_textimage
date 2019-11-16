import torch
import torch.nn as nn

# for pretrained model
import torchvision.models as models

import copy

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()  # the feature tensor of the style image

    def forward(self, input):
        target_gram = self._gram_matrix(self.target)
        input_gram = self._gram_matrix(input)
        loss_function = nn.MSELoss()
        self.loss = loss_function(input_gram, target_gram)

        return input

    def _gram_matrix(self, input):
        a, b, c, d = input.shape
        input = input.view(input.shape[0] * input.shape[1], -1)
        G = torch.mm(input, input.t())

        return G/(a*b*c*d)


def transfer_model(pretrained_model, content_img):
    pretrained_model = copy.deepcopy(pretrained_model)
    content_losses = []
    model = nn.Sequential()
    
    i = 0
    for layer in pretrained_model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))
        print("name: ", name)
        model.add_module(name, layer)
        if name in ['conv_1', 'conv_3', 'conv_5', 'conv_8', 'conv_11']:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss):
            break
        
    model = model[:(i+1)]

    return model, content_losses

def get_layer_info (pretrained_model, content_img):
    pretrained_model = copy.deepcopy(pretrained_model)



# model, content_losses = transfer_model(pt, style_img)
# model(content_img)
# content_score = 0
# for i, sl in enumerate(content_losses):
#     content_score += 100 * sl.loss
# content_score