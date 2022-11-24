import torch
import os
from .utils import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from pathlib import Path
from skimage.io import imread


class BFandSHG(Dataset):
    def __init__(self, mod1='BF', mod2='SHG', crop_size=225):
        self.crop_size = crop_size
        self.mod1 = list(Path('../ToCompare/DATA/train').glob('*_'+mod1+'.tif'))
        self.mod2 = list(Path('../ToCompare/DATA/train').glob('*_'+mod2+'.tif'))
        self.mod2.sort()
        self.mod1.sort()
     #   print(self.mod1)
     #   print(self.mod2)
        self.testmod1 = list(Path('../ToCompare/DATA/validation1').glob('*_'+mod1+'.tif'))
        self.testmod2 = list(Path('../ToCompare/DATA/validation1').glob('*_'+mod2+'.tif'))
        self.testmod1.sort()
        self.testmod2.sort()
     #   print(self.testmod1)
     #   print(self.testmod2)
        self.L = len(self.mod1)
        self.crop = crop_gen(self.crop_size)
        #TODO: if needed, do resizing to crop size, not cropping, as resizing actually preserves structures!
        self.resize = transforms.Resize(256) 
        self.to_tensor = transforms.ToTensor()

        #mod2 is anchor mod:
        #with to_tensor, channels already get moved to 0th axis, so no permuting and unsqueezing needed
        if mod2=='BF': #RGB
            #self.anchor_op = lambda x: x.permute((2,0,1)) #permute if needed
            #self.posneg_op = lambda x: x.unsqueeze(0).repeat(3,1,1)
            self.anchor_op = lambda x: x 
            self.posneg_op = lambda x: x.repeat(3,1,1)
        else:
            #self.anchor_op = lambda x: x.unsqueeze(0).repeat(3,1,1)
            #self.posneg_op = lambda x: x.permute((2,0,1))
            self.posneg_op = lambda x: x 
            self.anchor_op = lambda x: x.repeat(3,1,1)
            
        self.prepare_test()
        
    def __len__(self):
        return self.L

    def __getitem__(self, index):
        skt, imgs, idx, attr = self.get_triplet(index)
        
        imgs = [randomflip(self.crop(img)) for img in imgs] #enforcing a sort of rotational equivariance again???
        skt = randomflip(self.crop(skt))

        return (skt, *imgs, idx, attr)

    def get_triplet(self, index):
        neg_id = random.choice([l for l in range(self.L) if l!=index])

        img = self.mod2[index]
        pos_img  = self.mod1[index]
        neg_img = self.mod1[neg_id]

        img = self.resize(self.to_tensor(imread(img)))
        pos_img = self.resize(self.to_tensor(imread(pos_img)))
        neg_img = self.resize(self.to_tensor(imread(neg_img)))
       # print(f"sizes: anchor {img.shape}, pos {pos_img.shape}, neg {neg_img.shape}")

        img = self.anchor_op(img)
        pos_img, neg_img = self.posneg_op(pos_img), self.posneg_op(neg_img)

        idxs = torch.LongTensor([index, index, neg_id])
        attrs = 0
        return img, [pos_img, neg_img], idxs, attrs

    def prepare_test(self):
        self.testimgs1 = torch.stack([self.posneg_op(self.resize(self.to_tensor(imread(img)))) for img in self.testmod1])
        self.testimgs2 = torch.stack([self.anchor_op(self.resize(self.to_tensor(imread(img)))) for img in self.testmod2])
        self.test_idxs = torch.LongTensor([i for i in range(len(self.testmod1))])

    def get_test(self, complex=False):
        if not complex:
            return (self.crop(self.testimgs1, 'center'), self.crop(self.testimgs2, 'center'), self.test_idxs)
        elif complex and hasattr(self, 'test_data_complex'):
            return self.test_data_complex

        skts = []
        imgs = []
        for mode in ['center', 'upleft', 'upright', 'downleft', 'downright']:
            skts.append(self.crop(self.testimgs1, mode))
            imgs.append(self.crop(self.testimgs2, mode))

        skts = torch.cat(skts)
        imgs = torch.cat(imgs)
        skts = torch.cat([skts, randomflip(skts, p=1)], dim=0)
        imgs = torch.cat([imgs, randomflip(imgs, p=1)], dim=0)

        self.test_data_complex = (skts, imgs, self.test_idxs)
        return self.test_data_complex


    def loader(self, **args):
        return DataLoader(dataset=self, **args)
