# PIDNet (Version3)
Segment 11 classes 

## Virtual Environment 
```
python3.8 -m venv pidnet_env
```

# Preparation 

0. Clone the current repo and check to this branch 
```
git clone https://github.com/zlin-monarch/PIDNet.git
git checkout dev_11_classes
```

1. Modify `{project}/configs/constants.py` and configuration file 

2. (can be skipped) Split dataset and generating data list 
```
python3.8 split_data/create_list.py 
python3.8 check_dataset.py 
```

3. Donwload pretrained model 
```
cd pretrained_models/cityscapes
gdown https://drive.google.com/u/0/uc?id=1AR8LHC3613EKwG23JdApfTGsyOAcH0_L&export=download
```

# Reproduce the 20231009 Model Result 
- [Details to Train & Test 11 Segmentation Model](./docs/details_20231009.md)


# Training
## 1. Training

* Download the pretrained cityscapes models (PIDNet Large) and put them into `pretrained_models/Mavis/` dir.
* Edit configs/Mavis/{xxx}.yaml to update various training parameters.
````bash
python3.8 tools/train_tc.py --cfg_file_path configs/Mavis/{xxx}.yaml
````
or after editing args in `train_tc.sh`
```
Bash train.sh
```

## 2. Evaluation 
* Evaluate from images 
```bash
python3.8 tools/val_miou.py --cfg_file_path configs/Mavis/{xxx}.yaml
```
- Model: `config.TEST.MODEL_FILE` from `configs/Mavis/{xxx}.yaml`
- Image list: `config.DATASET.TEST_SET` from `configs/Mavis/{xxx}.yaml`

* Evaluate from video
```bash
python3.8 tools/test_video.py --input {video_file_path} --cfg_file_path configs/Mavis/{xxx}.yaml
```
or after editing args in `test_video.sh`
```
Bash test_video.sh
```
- Model: `config.TEST.MODEL_FILE` from `configs/Mavis/{xxx}.yaml`
- Video: `args.input`

# Error 
- Details: [Past Errors](./docs/error.md) 

# Refer
- [PIDNet Github](https://github.com/XuJiacong/PIDNet)
