#%%
# 
import torch
from torchvision import transforms
import numpy as np
from pathlib import Path
from skimage.io import imread
from data_loader.utils import *
import matplotlib.pyplot as plt

#%%
mod1 = list(Path('../ToCompare/DATA/train').glob('*_BF.tif'))
mod2 = list(Path('../ToCompare/DATA/train').glob('*_SHG.tif'))
crop_size = 225

anchor_op = lambda x: x.permute((2,0,1)) #permute if needed
posneg_op = lambda x: x.unsqueeze(0).repeat(3,1,1)
resize = transforms.Resize(256) #<- really needed? makes lower resolution imges
to_tensor = transforms.ToTensor()
crop = crop_gen(crop_size)

#%%
plt.figure()
img = mod2[0]
print(img)
img = imread(img)
print(f"read img: {img.shape}")
plt.subplot(1,4,1)
plt.imshow(img)
img = resize(to_tensor(img))
print(f"after to_tensor(resize): {img.shape}")
plt.subplot(1,4,2)
plt.imshow(img.permute((1,2,0)) )
img = crop(img)
print(f"after crop: {img.shape}")
plt.subplot(1,4,3)
plt.imshow(img.permute((1,2,0)) )
img = randomflip(img)
print(f"after flip: {img.shape}")
plt.subplot(1,4,4)
plt.imshow(img.permute((1,2,0)) )

# %%
import pandas as pd

epc = 100
fajl = 'SHG_to_BF.txt'
p = Path('logs',  fajl)
toskip = [x+i*14 for i in range(epc//10) for x in [10,11,12,13]] 
dtf = pd.read_csv(p, skiprows=toskip, header=None)
#dtf

getnr = lambda xind: [float(x.split(':')[1]) for x in dtf.iloc[:,xind]]

outs = {'triplet': getnr(1), 'sphere': getnr(2), 'sofmax': getnr(4), 'LOSS': getnr(5)}
# 'centre': getnr(3),
plt.figure()
for k,v in outs.items():
    plt.plot(range(epc), v, label=k)
plt.legend()
plt.show()

# %%
