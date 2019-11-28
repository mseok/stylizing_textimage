import torch
import torchvision
# import matplotlib as mpl
# import matplotlib.pylab as plt
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import argparse


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


# ysDataset
# return dictionary with source and glyph
# which shape is 64*(64*26)*3 numpy array
class ysDataset(Dataset):
    def __init__(self, png_floder, transform=None):
        self.png_list = glob.glob(png_floder + '/**/*.png', recursive=True)
        self.transform = transform

    def __len__(self):
        return len(self.png_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = plt.imread(self.png_list [idx])
        data_gray_address = self.png_list[idx].replace("_colorGrad64", '64')[:-5] + '0.png'
        data_gray = plt.imread(data_gray_address)
    
        if (len(data_gray.shape) == 2):
            data_gray = np.stack((data_gray,)*3, axis=-1) # gray -> rgb

        if (len(data.shape) == 2):
            data = np.stack((data,)*3, axis=-1) # gray -> rgb


        # data_tensor = torch.from_numpy(data).permute(2,0,1) # 3*64*(64*26)
        # data_gray_tensor = torch.from_numpy(data_gray).permute(2,0,1) # 3*64*(64*26)

        # sample = {'source': data_tensor, 'glyph': data_gray_tensor}
        sample = {'source': data, 'glyph': data_gray} # 64*(64*26)*3

        if self.transform:
            sample = self.transform(sample)

        return sample

def load_dataset_with_glyph (args):
    ys_data = ysDataset (args.color_path)
    # print (args.color_path + '/*.png')

    ys_dataloader = DataLoader (
                        ys_data, 
                        batch_size=args.batch_size,
                        num_workers=1,
                        shuffle=True)

    return ys_dataloader


# 사용 예시.
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.color_path = 'datasets/Capitals_colorGrad64'
    args.batch_size = 10

    print (args.batch_size)
    for i, sample_batched in enumerate(load_dataset_with_glyph (args)):
        print (len(sample_batched))
        sample_batched['source'] = sample_batched['source'][:1]
        print (sample_batched['source'][0].type())
        print(i, sample_batched['source'].permute(0,3,1,2).shape, sample_batched['glyph'].shape)
        if i==2:
            plt.imshow (np.concatenate([sample_batched['source'][0], sample_batched['glyph'][0]],0))
            plt.show()
            break


