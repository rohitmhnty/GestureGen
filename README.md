



# GestureGen


# ⚒️ Installation

## Build Environtment

```
conda create -n gesturelsm python=3.12
conda activate gesturelsm
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
bash demo/install_mfa.sh
```


## Download Model (1-speaker and all-speaker)
```
# From Google Drive
# Download the pretrained model (Shortcut) + (Shortcut-reflow) + (Diffusion) + (RVQ-VAEs)
gdown https://drive.google.com/drive/folders/1OfYWWJbaXal6q7LttQlYKWAy0KTwkPRw?usp=drive_link -O ./ckpt --folder


# Download the SMPL model
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
# Evaluate the pretrained shortcut model (20 steps)
python test.py -c configs/shortcut_rvqvae_128.yaml

# Evaluate the pretrained shortcut-reflow model (2-step)
python test.py -c configs/shortcut_reflow_test.yaml

# Evaluate the pretrained diffusion model
python test.py -c configs/diffuser_rvqvae_128.yaml

```

## Train RVQ-VAEs 
> Require download dataset 
```
bash train_rvq.sh
```

## Train Generator 
> Require download dataset 
```

# Train the shortcut model
python train.py -c configs/shortcut_rvqvae_128.yaml

# Train the diffusion model
python train.py -c configs/diffuser_rvqvae_128.yaml
```


## Demo 
```
python demo.py -c configs/shortcut_rvqvae_128_hf.yaml
```



#  Acknowledgments
Thanks to [SynTalker](https://github.com/RobinWitch/SynTalker/tree/main), [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024), [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture), our code is partially borrowing from them. Please check these useful repos.
