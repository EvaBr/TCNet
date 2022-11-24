#%%
from pathlib import Path 
from models import *
import torch
from torch.backends import cudnn
from data_loader.utils import *
from torchvision import transforms
from skimage.io import imread

crop_size = 256
resize_size = 225
modl = 'BF_to_SHG'
modl = 'SHG_to_BF'



device = torch.device('cuda:0')
cudnn.benchmark = True

crop = crop_gen(crop_size)
resize = lambda x: x #transforms.Resize(resize_size)  #use resize2 instead, when doing patches (so that cropping is done first!)
resize2 = lambda x: x #transforms.Resize(resize_size) 
to_tensor = transforms.ToTensor()


model = DenseNet(f_dim=1024, norm=False)
model.to(device)
model_path = f'logs/models/{modl}_100.pth'
model.load_state_dict(torch.load(model_path))
model.eval()



#modl = 'SHG_to_BF'
spl = modl.split('_')
mod1 = spl[-1]
mod2 = spl[0] #was used as anchor during training and best results should be when this is the database modlaity
# but anyway run test with both directions!

#%%

def get_features(model, imgs):
    with torch.no_grad():
        N = len(imgs)
        feats = []
        for i in range(N):
            img = imgs[i,...].unsqueeze(0)
            feats.append(model(img.to(device)).cpu())
   # print(img.shape)
   # print(feats[0].shape)
    stacked = torch.cat(feats)
   # print(stacked.shape)
    return stacked

#%%
print(modl)
mod2 = 'SHG'
mod1 = 'BF'

mods1 = list(Path('../ToCompare/DATA/test').glob('R*_'+mod1+'.tif'))
mods2R = list(Path('../ToCompare/DATA/test').glob('R*_'+mod2+'.tif'))
mods2T = list(Path('../ToCompare/DATA/test').glob('T*_'+mod2+'.tif'))
mods1.sort()
mods2R.sort()
mods2T.sort()

mod1_op = lambda x: x
mod2_op = lambda x: x
if mod1=='SHG': 
    mod1_op = lambda x: x.repeat(3,1,1)
if mod2=='SHG':
    mod2_op = lambda x: x.repeat(3,1,1)


imgs1 = torch.stack([mod1_op(resize(to_tensor(imread(img)))) for img in mods1])
imgs2R = torch.stack([mod2_op(resize(to_tensor(imread(img)))) for img in mods2R])
imgs2T = torch.stack([mod2_op(resize(to_tensor(imread(img)))) for img in mods2T])
test_idxs = torch.LongTensor([i for i in range(len(imgs1))])

#imgs1 = crop(imgs1, 'center')
imgs2R = crop(imgs2R, 'center')
imgs2T = crop(imgs2T, 'center')
imgs1 = resize2(imgs1)
imgs2R = resize2(imgs2R)
imgs2T = resize2(imgs2T)

ns = len(mods1)

print(f"Searching for {mod2} in {mod1}:")
with torch.no_grad():
    all_mod1_feats = get_features(model, imgs1) #model(imgs1.to(device)).cpu()
    all_mod2R_feats = get_features(model, imgs2R) #model(imgs2R.to(device)).cpu()
    all_mod2T_feats = get_features(model, imgs2T) #model(imgs2T.to(device)).cpu()


    top1R, top1T = 0, 0
    top5R, top5T = 0, 0
    top10R, top10T = 0, 0
    for index in range(ns):
        queryR = all_mod2R_feats[index,...].unsqueeze(0).repeat(ns, 1) 
        queryT = all_mod2T_feats[index,...].unsqueeze(0).repeat(ns, 1) 

        resR = torch.norm(queryR - all_mod1_feats, dim=1).pow(2)
        resT = torch.norm(queryT - all_mod1_feats, dim=1).pow(2)
        retrieval_idxsR = resR.sort()[1][:10]
        retrieval_idxsT = resT.sort()[1][:10]

        test_idxs = test_idxs.type(retrieval_idxsR.type())

        foundR = [1 if index==retrieval_idx else 0 for retrieval_idx in retrieval_idxsR]
        foundT = [1 if index==retrieval_idx else 0 for retrieval_idx in retrieval_idxsT]

        top10T += sum(foundT[:10])
        top5T +=  sum(foundT[:5])
        top1T += foundT[0]
        top10R += sum(foundR[:10])
        top5R +=  sum(foundR[:5])
        top1R += foundR[0]

    top1R, top1T = top1R/ns, top1T/ns
    top5R, top5T = top5R/ns, top5T/ns
    top10R, top10T = top10R/ns, top10T/ns
    print(f"(R) top-1: {top1R:.03f}, top-5: {top5R:.03f}, top-10: {top10R:.03f}")
    print(f"(T) top-1: {top1T:.03f}, top-5: {top5T:.03f}, top-10: {top10T:.03f}")
print()




    #mod1_feats = all_mod1_feats.unsqueeze(1).repeat(ns, 1, 1)
    #mod2R_feats = all_mod2R_feats.unsqueeze(0).repeat(1, ns, 1) #query
    #mod2T_feats = all_mod2T_feats.unsqueeze(0).repeat(1, ns, 1) #query
        
    #resR = torch.norm(mod2R_feats - mod1_feats, dim=2).pow(2)
    #resT = torch.norm(mod2T_feats - mod1_feats, dim=2).pow(2)
    #retrieval_idxsR = resR.sort(dim=1)[1][:,:10]
    #retrieval_idxsT = resT.sort(dim=1)[1][:,:10]
   
    #test_idxs = test_idxs.type(retrieval_idxsR.type())

    #for retrieval_idxs,tip in zip([retrieval_idxsR, retrieval_idxsT], ['R', 'T']):
    #    accu1 = (retrieval_idxs[:,0] == test_idxs).float().mean().item()
    #    accu10 = accu1
    #    accu5 = accu1
    #    for i in range(1, 10):
    #        accu = (retrieval_idxs[:,i] == test_idxs).float().mean().item()
    #        accu10 += accu
    #    for i in range(1, 5):
    #        accu = (retrieval_idxs[:,i] == test_idxs).float().mean().item()
    #        accu5 += accu
    #    print(f"({tip}) top-1: {accu1}, top-5: {accu5}, top-10: {accu10}")
    #print()

# %%
#what would random retrieval be in numbers:
h = 134 #268
top1, top5, top10 = 0, 0, 0
avg = 1000
for a in range(avg):
    for k in range(h):
        match = np.random.permutation(h)
        if match[0]==k:
            top1+=1
        if k in match[:5]:
            top5+=1
        if k in match[:10]:
            top10+=1

print(f"averaged over {avg} \n top1: {top1/(avg*h):.4f}, top5: {top5/(avg*h):.4f}, top10: {top10/(avg*h):.4f}")
# %%
