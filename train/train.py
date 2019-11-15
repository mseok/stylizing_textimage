# for pretrained model
import torchvision.models as models


pt = models.vgg16(pretrained=True).features.eval()
