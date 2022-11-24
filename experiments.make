

MODEL=--model_type=densenet --feat_dim=1024 
LOSS=--loss_type='triplet,softmax,sphere,centre' --loss_ratio='0.15,0.3,0.2,0.0003' 
DATALOADER=--obj=bf_shg

MOD1=BF 
MOD2=SHG
FL=_noCrop

#CROP=--crop_size=225
CROP=--crop_size=0
EPCH=--nr_epochs=100 --batch_size=1

all: train test 

train:
	python train.py $(MODEL) $(LOSS) $(DATALOADER) $(CROP) --flag=SHG_to_BF$(FL) --mod1=BF --mod2=SHG --epoch_sep=10 $(EPCH)
	python train.py $(MODEL) $(LOSS) $(DATALOADER) $(CROP) --flag=BF_to_SHG$(FL) --mod1=SHG --mod2=BF --epoch_sep=10 $(EPCH)

test: