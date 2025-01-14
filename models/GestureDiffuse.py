import time
import inspect
import logging
from typing import Optional

import tqdm
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from models.config import instantiate_from_config
from models.utils.utils import count_parameters, extract_into_tensor, sum_flat

logger = logging.getLogger(__name__)


class GestureDiffusion(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.modality_encoder = instantiate_from_config(cfg.model.modality_encoder)
        self.denoiser = instantiate_from_config(cfg.model.denoiser)
        self.scheduler = instantiate_from_config(cfg.model.scheduler)
        self.alphas = torch.sqrt(self.scheduler.alphas_cumprod)
        self.sigmas = torch.sqrt(1 - self.scheduler.alphas_cumprod)

        self.do_classifier_free_guidance = cfg.model.do_classifier_free_guidance
        self.guidance_scale = cfg.model.guidance_scale
        self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduction='none')

    def summarize_parameters(self) -> None:
        logger.info(f'Denoiser: {count_parameters(self.denoiser)}M')
        logger.info(f'Scheduler: {count_parameters(self.modality_encoder)}M')
    


    def predicted_origin(self, model_output: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> tuple:
        self.alphas = self.alphas.to(model_output.device)
        self.sigmas = self.sigmas.to(model_output.device)
        alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
        sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)

        # i will do this
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sigmas * model_output) / alphas
            pred_epsilon = model_output
        
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alphas * model_output) / sigmas
        
        elif self.scheduler.config.prediction_type == "v_prediction":
            sigmas = extract_into_tensor(self.sigmas, timesteps, sample.shape)
            alphas = extract_into_tensor(self.alphas, timesteps, sample.shape)
            pred_original_sample = alphas * sample - sigmas * model_output
        else:
            raise ValueError(f"Invalid prediction_type {self.scheduler.config.prediction_type}.")

        return pred_original_sample, pred_epsilon



    def forward(self, cond_: dict) -> dict:

        audio = cond_['y']['audio']
        word = cond_['y']['word']
        id = cond_['y']['id']
        seed = cond_['y']['seed']
        mask = cond_['y']['mask']
        style_feature = cond_['y']['style_feature']
        wavlm_feat = cond_['y']['wavlm']
        
        audio_feat = self.modality_encoder(audio, word, wavlm_feat)

        bs = audio_feat.shape[0]
        shape_ = (bs, 128 * 3, 1, 32)
        latents = torch.randn(shape_, device=audio_feat.device)

        latents = self._diffusion_reverse(latents, seed, audio_feat, guidance_scale=self.guidance_scale)

        return latents



    def _diffusion_reverse(
            self,
            latents: torch.Tensor,
            seed: torch.Tensor,
            at_feat: torch.Tensor,
            guidance_scale: float = 1,
    ) -> torch.Tensor:

        return_dict = {}
        # scale the initial noise by the standard deviation required by the scheduler, like in Stable Diffusion
        # this is the initial noise need to be returned for rectified training
        latents = latents * self.scheduler.init_noise_sigma

       
        noise = latents

        
        return_dict["init_noise"] = latents
        return_dict['at_feat'] = at_feat
        return_dict['seed'] = seed

        # set timesteps
        self.scheduler.set_timesteps(self.cfg.model.scheduler.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(at_feat.device)

        latents = torch.zeros_like(latents)
        
        latents = self.scheduler.add_noise(latents, noise, timesteps[0])
        
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.cfg.model.scheduler.eta

        for i, t in tqdm.tqdm(enumerate(timesteps)):
            latent_model_input = latents
            # actually it does nothing here according to ddim scheduler
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            model_output = self.denoiser.forward_with_cfg(
                x=latent_model_input,
                timesteps=t,
                seed=seed,
                at_feat=at_feat,
                guidance_scale=guidance_scale)

            latents = self.scheduler.step(model_output, t, latents, **extra_step_kwargs).prev_sample
        return_dict['latents'] = latents
        return return_dict