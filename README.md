[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gesturelsm-latent-shortcut-based-co-speech/gesture-generation-on-beat2)](https://paperswithcode.com/sota/gesture-generation-on-beat2?p=gesturelsm-latent-shortcut-based-co-speech) <a href="https://arxiv.org/abs/2501.18898"><img src="https://img.shields.io/badge/arxiv-gray?logo=arxiv&amp"></a>



# GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling


# 📝 Release Plans

- [x] Inference Code
- [x] Pretrained Models
- [x] A web demo
- [x] Training Code

# ⚒️ Installation

## Build Environtment

```
conda create -n gesturelsm python=3.12
conda activate gesturelsm
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
bash demo/install_mfa.sh
```

## Results

![Beat Results](beat-new.png)

This table shows the results of 1-speaker and all-speaker comparisons. RAG-Gesture refers to [**Retrieving Semantics from the Deep: an RAG Solution for Gesture Synthesis**](https://arxiv.org/abs/2412.06786), accepted by CVPR 2025.

## Note
- The statistics reported in the paper is based on 1-speaker with speaker-id of 2, 'scott' in order to be consistent with the previous SOTA methods.
- The pretrained model previously provided is trained on 1-speaker. (RVQ-VAEs, Diffusion, Shortcut, Shortcut-reflow)
- If you want to use all-speaker, please modify the config files to include all speaker ids.
- April 16, 2025: update the pretrained model to include all speakers. (RVQ-VAEs, Shortcut)
- I did not do hyperparameter tuning for all-speaker, but just use the same setting as 1-speaker.
- You can add the speaker embedding if you want to have a better performance for all-speakers. I did not add it to make the code more aligned with the current 1-speaker setting and make the model capable of generating gesture for a novel speaker.
- No gesture type information is used in the current version. The reason is that for a novel speaker and novel setting, you never know the gesture type. As a result, including this information is not realistic for real-world applications. However, if you just want to see a even better FGD, you can try to add this information.





## Download Model (1-speaker and all-speaker)
```
# Option 1: From Google Drive
# Download the pretrained model (Shortcut) + (Shortcut-reflow) + (Diffusion) + (RVQ-VAEs)
gdown https://drive.google.com/drive/folders/1OfYWWJbaXal6q7LttQlYKWAy0KTwkPRw?usp=drive_link -O ./ckpt --folder

# Option 2: From Huggingface Hub
huggingface-cli download https://huggingface.co/pliu23/GestureLSM --local-dir ./ckpt

# Download the SMPL model
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./datasets/hub --folder
```

## Download Dataset
> For evaluation and training, not necessary for running a web demo or inference.

- Download the original raw data
```
bash preprocess/bash_raw_cospeech_download.sh
```

## Eval (1-speaker)
> Require download dataset 
```
# Evaluate the pretrained shortcut model (20 steps)
python test.py -c configs/shortcut_rvqvae_128.yaml

# Evaluate the pretrained shortcut-reflow model (2-step)
python test.py -c configs/shortcut_reflow_test.yaml

# Evaluate the pretrained diffusion model
python test.py -c configs/diffuser_rvqvae_128.yaml

```

## Train RVQ-VAEs (1-speaker)
> Require download dataset 
```
bash train_rvq.sh
```

## Train Generator (1-speaker)
> Require download dataset 
```

# Train the shortcut model
python train.py -c configs/shortcut_rvqvae_128.yaml

# Train the diffusion model
python train.py -c configs/diffuser_rvqvae_128.yaml
```


## Demo (1-speaker)
```
python demo.py -c configs/shortcut_rvqvae_128_hf.yaml
```



# 🙏 Acknowledgments
Thanks to [SynTalker](https://github.com/RobinWitch/SynTalker/tree/main), [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024), [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture), our code is partially borrowing from them. Please check these useful repos.


# 📖 Citation

If you find our code or paper helps, please consider citing:

```bibtex
@misc{liu2025gesturelsmlatentshortcutbased,
      title={GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling}, 
      author={Pinxin Liu and Luchuan Song and Junhua Huang and Haiyang Liu and Chenliang Xu},
      year={2025},
      eprint={2501.18898},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.18898}, 
}
```