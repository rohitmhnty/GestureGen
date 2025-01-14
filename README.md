# GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling


# 📝 Release Plans

- [x] Inference Code
- [ ] Pretrained Models
- [ ] A web demo
- [ ] Training Code

# ⚒️ Installation

## Build Environtment

```
conda create -n gesturelsm python=3.12
conda activate gesturelsm
pip install -r requirements.txt
bash demo/install_mfa.sh
```

## Download Model
```
gdown https://drive.google.com/drive/folders/1tGTB40jF7v0RBXYU-VGRDsDOZp__Gd0_?usp=drive_link -O ./ckpt --folder
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./datasets/hub --folder
```

## Download Dataset
> For evaluation and training, not necessary for running a web demo or inference.

- Download the original raw data
```
bash preprocess/bash_raw_cospeech_download.sh
```

## Eval
> Require download dataset 
```
python test.py -c configs/shortcut_rvqvae_128.yaml
```