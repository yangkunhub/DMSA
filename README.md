# DMSA: Dynamic Multi-scale Unsupervised Semantic Segmentation Based on Adaptive Affinity
[Kun Yang](https://github.com/yangkunhub),


[[Preprint](https://arxiv.org/abs/2303.00199)]

# create conda env
conda env create -f environment.yaml
# activate conda env
conda activate DMSA

```

## Dataset Preparation

- MS-COCO Dataset: Download the [trainset](http://images.cocodataset.org/zips/train2017.zip), [validset](http://images.cocodataset.org/zips/val2017.zip), [annotations](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip) and the [json files](http://images.cocodataset.org/annotations/annotations_trainval2017.zip), place the extracted files into `root/data/MSCOCO`.

- PascalVOC Dataset: Download [training/validation data](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar), place the extracted files into `root/data/PascalVOC`.

- SBD Dataset: Download [training/validation data](http://home.bharathh.info/pubs/codes/SBD/download.html), place the extracted files into `root/data/SBD`.


## PascalVOC_AUG dataset making



the structure of dataset folders should be as follow:
~~~

data/
    │── MSCOCO/
    │     ├── images/
    │     │     ├── train2017/
    │     │     └── val2017/
    │     └── annotations/
    │           ├── train2017/
    │           ├── val2017/
    │           ├── instances_train2017.json
    │           └── instances_val2017.json
    │
    │── PascalVOC/
    │     ├── JPEGImages/
    │     ├── SegmentationClass/
    │     └── ImageSets/
    │           └── Segmentation/
    │                   ├── train.txt
    │                   └── val.txt
    │── PascalVOC_aug/
         ├── JPEGImages/
         ├── SegmentationClass/
         └── ImageSets/
               └── Segmentation/
                       ├── train.txt
                       └── val.txt


## Model download
- please download the pretrained [dino model (deit small 8x8/16*16)](https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth), then place it into `root/weight/dino/` 
- download trained model from [Google Drive](https://drive.google.com/drive/folders/1vHKLrAE51mLTK-5DpzByQ_g1RAjmONyi?usp=sharing) or [Baidu Netdisk (code:1118)](https://pan.baidu.com/s/1N7GSzcMOi9C3mgpUsIa4oA), then place them into `root/weight/trained/` 

#Run and evaluation(Note that you need to use the pre-generated initial pseudo-mask)

# MSCOCO 80
#  train model
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python coco_trainval.py -F train/coco with dataset.num_workers=32 model.decoder.n_things=80
```
#  evaluate COCO-80:
```bash
CUDA_VISIBLE_DEVICES=0 python coco_trainval.py -F eval/coco_80 with eval_only=1 model.decoder.n_things=80 model.decoder.pretrained_weight=train/coco/(number)/decoder_best.pt
```

#  PascalVOC 2012
#  train model
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python pascalvoc_trainval.py -F train/pascalvoc with dataset.num_workers=32
```
#  evaluate PascalVOC 2012
```bash
CUDA_VISIBLE_DEVICES=0 python pascalvoc_trainval.py -F eval/pascalvoc with eval_only=1 model.decoder.n_things=20 model.decoder.pretrained_weight=train/pascalvoc/(number)/decoder_best.pt
```

#  PascalVOC 2012 AUG
#  train model
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python pascalvoc_aug_trainval.py -F train/pascalvoc with dataset.num_workers=32
```
#  evaluate PascalVOC 2012 AUG
```bash
CUDA_VISIBLE_DEVICES=0 python pascalvoc_aug_trainval.py -F eval/pascalvoc with eval_only=1 model.decoder.n_things=20 model.decoder.pretrained_weight=train/pascalvoc_AUG/(number)/decoder_best.pt
```
