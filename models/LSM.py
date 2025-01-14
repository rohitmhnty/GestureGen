import time
import inspect
import logging
from typing import Optional

import scipy.stats as stats
import tqdm
import numpy as np
from omegaconf import DictConfig
from typing import Dict
import math

import torch
import torch.nn.functional as F
from models.config import instantiate_from_config
from models.utils.utils import count_parameters, extract_into_tensor, sum_flat

logger = logging.getLogger(__name__)

def exponential_pdf(x, a):
    C = a / (np.exp(a) - 1)
    return C * np.exp(a * x)

# Define a custom probability density function
class ExponentialPDF(stats.rv_continuous):
    def _pdf(self, x, a):
        return exponential_pdf(x, a)

def sample_t(exponential_pdf, num_samples, a=2):
    t = exponential_pdf.rvs(size=num_samples, a=a)
    t = torch.from_numpy(t).float()
    t = torch.cat([t, 1 - t], dim=0)
    t = t[torch.randperm(t.shape[0])]
    t = t[:num_samples]

    t_min = 1e-5
    t_max = 1-1e-5

    # Scale t to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    return t

def sample_t_fast(num_samples, a=2, t_min=1e-5, t_max=1-1e-5):
    # Direct inverse sampling for exponential distribution
    C = a / (np.exp(a) - 1)
    
    # Generate uniform samples
    u = torch.rand(num_samples * 2)
    
    # Inverse transform sampling formula for the exponential PDF
    # F^(-1)(u) = (1/a) * ln(1 + u*(exp(a) - 1))
    t = (1/a) * torch.log(1 + u * (np.exp(a) - 1))
    
    # Combine t and 1-t
    t = torch.cat([t, 1 - t])
    
    # Random permutation and slice
    t = t[torch.randperm(t.shape[0])][:num_samples]
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def sample_cosmap(num_samples, t_min=1e-5, t_max=1-1e-5, device='cpu'):
    """
    CosMap sampling.
    Args:
        num_samples: Number of samples to generate
        t_min, t_max: Range limits to avoid numerical issues
    """
    # Generate uniform samples
    u = torch.rand(num_samples, device=device)
    
    # Apply the cosine mapping
    pi_half = torch.pi / 2
    t = 1 - 1 / (torch.tan(pi_half * u) + 1)
    
    # Scale to [t_min, t_max]
    t = t * (t_max - t_min) + t_min
    
    return t

def reshape_coefs(t):
    return t.reshape((t.shape[0], 1, 1, 1))

class GestureLSM(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        # Initialize model components
        self.modality_encoder = instantiate_from_config(cfg.model.modality_encoder)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)

        # Model hyperparameters
        self.do_classifier_free_guidance = cfg.model.do_classifier_free_guidance
        self.guidance_scale = cfg.model.guidance_scale
        self.num_inference_steps = cfg.model.n_steps
        self.exponential_distribution = ExponentialPDF(a=0, b=1, name='ExponentialPDF')

        # Loss functions
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')

    def summarize_parameters(self) -> None:
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')
        logger.info(f'Encoder: {count_parameters(self.modality_encoder)}M')
    
    def forward(self, condition_dict: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
        """Forward pass for inference.
        
        Args:
            condition_dict: Dictionary containing input conditions including audio, word tokens,
                          and other features
        
        Returns:
            Dictionary containing generated latents
        """
        # Extract input features
        audio = condition_dict['y']['audio']
        word_tokens = condition_dict['y']['word']
        ids = condition_dict['y']['id']
        seed_vectors = condition_dict['y']['seed']
        mask = condition_dict['y']['mask']
        style_features = condition_dict['y']['style_feature']
        wavlm_features = condition_dict['y']['wavlm']
        
        # Encode input modalities
        audio_features = self.modality_encoder(audio, word_tokens, wavlm_features)

        # Initialize generation
        batch_size = audio_features.shape[0]
        latent_shape = (batch_size, 128 * 3, 1, 32)
        
        # Sampling parameters
        x_t = torch.randn(latent_shape, device=audio_features.device)
        epsilon = 1e-8
        delta_t = torch.tensor(1 / self.num_inference_steps).to(audio_features.device)
        timesteps = torch.linspace(epsilon, 1 - epsilon, self.num_inference_steps + 1).to(audio_features.device)
        
        # Generation loop
        for step in range(1, len(timesteps)):
            current_t = timesteps[step - 1].unsqueeze(0).repeat((batch_size,))
            current_delta = delta_t.unsqueeze(0).repeat((batch_size,))
            
            with torch.no_grad():
                speed = self.denoiser.forward_with_cfg(
                    x=x_t,
                    timesteps=current_t,
                    seed=seed_vectors,
                    at_feat=audio_features,
                    cond_time=current_delta,
                    guidance_scale=self.guidance_scale
                )
               
            x_t = x_t + (timesteps[step] - timesteps[step - 1]) * speed

        return {'latents': x_t}