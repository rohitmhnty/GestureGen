[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/gesturelsm-latent-shortcut-based-co-speech/gesture-generation-on-beat2)](https://paperswithcode.com/sota/gesture-generation-on-beat2?p=gesturelsm-latent-shortcut-based-co-speech) <a href="https://arxiv.org/abs/2501.18898"><img src="https://img.shields.io/badge/arxiv-gray?logo=arxiv&amp"></a>



# GestureLSM: Latent Shortcut based Co-Speech Gesture Generation with Spatial-Temporal Modeling


# 📝 Release Plans

- [x] Inference Code
- [x] Pretrained Models
- [x] A web demo
- [ ] Training Code

# ⚒️ Installation

## Build Environtment

```
conda create -n gesturelsm python=3.12
conda activate gesturelsm
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
bash demo/install_mfa.sh
```

## Download Model
```
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

## Demo
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
      author={Pinxin Liu and Luchuan Song and Junhua Huang and Chenliang Xu},
      year={2025},
      eprint={2501.18898},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.18898}, 
}
```