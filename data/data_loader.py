import torch
import torchvision
# import matplotlib as mpl
# import matplotlib.pylab as plt
import os

# alphabet_position
# 영어 text로 주면 그 영어 알파벳 각각을 indexing 한 리스트를 아웃풋으로 낸다.
# ex) alphalphabet_position ('google') = [7, 15, 15, 7, 12, 5]
def alphabet_position(text):
    nums = [str(ord(x) - 96) for x in text.lower() if x >= 'a' and x <= 'z']
    return list(map(int, nums))

# load_glyph_dataset
# glyph dataset 을 로드하여 DataLoader를 리턴한다.
# color=True 면 ramdom_colored 된 데이터를 가져온다.
# output 은 batch_size*3*64*(64*26) 의 tensor.
# A 부터 Z 까지 이어져 있으니 잘라서 사용하면 됨.
def load_dataset(args, color=False):
    
    if color:
        data_path = args.color_path
    else:
        data_path = args.noncolor_path
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader 

