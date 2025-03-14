import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .layers.utils import *
from .layers.transformer import SpatialTemporalBlock, CrossAttentionBlock

class GestureDenoiser(nn.Module):
    def __init__(self,
        input_dim=128,
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        n_seed=8,
        flip_sin_to_cos= True,
        freq_shift = 0,
        cond_proj_dim=None,
        use_exp=False,
    
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.use_exp = use_exp
        self.joint_num = 3 if not self.use_exp else 4
        
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(dim=self.latent_dim*self.joint_num,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(3)])
        
        self.mytimmblocks = nn.ModuleList([
            SpatialTemporalBlock(dim=self.latent_dim,num_heads=self.num_heads,mlp_ratio=self.ff_size//self.latent_dim,drop_path=self.dropout) #hidden是对应于输入x的维度，attn_heads应该是12，这里写1是为了方便调试流程
                for _ in range(self.num_layers)])
            
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed
        
        self.embed_text = nn.Linear(self.input_dim*self.joint_num*4, self.latent_dim)

        self.output_process = OutputProcess(self.input_dim, self.latent_dim)

        self.rel_pos = SinusoidalEmbeddings(self.latent_dim)
        self.input_process = InputProcess(self.input_dim , self.latent_dim)
        self.input_process2 = nn.Linear(self.latent_dim*2, self.latent_dim)
        
        self.time_embedding = TimestepEmbedding(self.latent_dim, self.latent_dim, self.activation, cond_proj_dim=cond_proj_dim, zero_init_cond=True)
        time_dim = self.latent_dim
        self.time_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        if cond_proj_dim is not None:
            self.cond_proj = Timesteps(time_dim, flip_sin_to_cos, freq_shift)
        
        self.null_cond_embed = nn.Parameter(torch.zeros(32, self.latent_dim*self.joint_num), requires_grad=True)

    # dropout mask
    def prob_mask_like(self, shape, prob, device):
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
    


    @torch.no_grad()
    def forward_with_cfg(self, x, timesteps, seed, at_feat, cond_time=None, guidance_scale=1):
        """
        Forward pass with classifier-free guidance.
        Args:
            x: [batch_size, njoints, nfeats, max_frames]
            timesteps: [batch_size]
            seed: the previous gesture segment
            at_feat: the audio feature
            guidance_scale: Scale for classifier-free guidance (1.0 means no guidance)
        """
        # Run both conditional and unconditional in a single forward pass
        if guidance_scale > 1:
            output = self.forward(
                x,
                timesteps,
                seed,
                at_feat,
                cond_time=cond_time,
                cond_drop_prob=0.0,
                null_cond=False,
                do_classifier_free_guidance=True
            )
            # Split predictions and apply guidance
            pred_cond, pred_uncond = output.chunk(2, dim=0)
            guided_output = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            
        else:
            guided_output = self.forward(x, timesteps, seed, at_feat, cond_time=cond_time, cond_drop_prob=0.0, null_cond=False)
        
        return guided_output
    


    def forward(self, x, timesteps, seed, at_feat, cond_time=None, cond_drop_prob: float = 0.1, null_cond=False, do_classifier_free_guidance=False, force_cfg=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        seed: [batch_size, njoints, nfeats]
        do_classifier_free_guidance: whether to perform classifier-free guidance (doubles batch)
        """
        _,_,_,noise_length = x.shape

        if x.shape[2] == 1:
            x = x.squeeze(2)
            x = x.reshape(x.shape[0], self.joint_num, -1, x.shape[2])
            
        # Double the batch for classifier free guidance
        if do_classifier_free_guidance and not self.training:
            x = torch.cat([x] * 2, dim=0)
            seed = torch.cat([seed] * 2, dim=0)
            at_feat = torch.cat([at_feat] * 2, dim=0)
       
        bs, njoints, nfeats, nframes = x.shape      # [bs, 3, 128, 32]
        
        # need to be an arrary, especially when bs is 1
        timesteps = timesteps.expand(bs).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=x.dtype)

        if cond_time is not None and self.cond_proj is not None:
            cond_time = cond_time.expand(bs).clone()
            cond_emb = self.cond_proj(cond_time)
            cond_emb = cond_emb.to(dtype=x.dtype)
            emb_t = self.time_embedding(time_emb, cond_emb)
        else:
            emb_t = self.time_embedding(time_emb)
        
        if self.n_seed != 0:
            embed_text = self.embed_text(seed.reshape(bs, -1))
            emb_seed = embed_text
        
        # Handle both conditional and unconditional branches in a single forward pass
        if do_classifier_free_guidance and not self.training:
            # First half of batch: conditional, Second half: unconditional
            null_cond_embed = self.null_cond_embed.to(at_feat.dtype)
            at_feat_uncond = null_cond_embed.unsqueeze(0).expand(bs//2, -1, -1)
            at_feat = torch.cat([at_feat[:bs//2], at_feat_uncond], dim=0)
        else:
            if force_cfg is None:
                if self.training:
                    keep_mask = self.prob_mask_like((bs,), 1 - cond_drop_prob, device=at_feat.device)
                    keep_mask_embed = rearrange(keep_mask, "b -> b 1 1")
                    
                    null_cond_embed = self.null_cond_embed.to(at_feat.dtype)
                    at_feat = torch.where(keep_mask_embed, at_feat, null_cond_embed)

                if null_cond:
                    at_feat = self.null_cond_embed.to(at_feat.dtype).unsqueeze(0).expand(bs, -1, -1)
            else:
                force_cfg = torch.tensor(force_cfg, device=at_feat.device)
                force_cfg_embed = rearrange(force_cfg, "b -> b 1 1")

                null_cond_embed = self.null_cond_embed.to(at_feat.dtype)
                at_feat = torch.where(force_cfg_embed, at_feat, null_cond_embed)

        
        xseq = self.input_process(x)

        # add the seed information
        embed_style_2 = (emb_seed + emb_t).unsqueeze(1).unsqueeze(2).expand(-1, self.joint_num, 32, -1)  # (300, 256)
        xseq = torch.cat([embed_style_2, xseq], axis=-1)  # -> [88, 300, 576]
        
        xseq = self.input_process2(xseq)
        

        # apply the positional encoding
        xseq = xseq.reshape(bs * self.joint_num, nframes, -1)
        pos_emb = self.rel_pos(xseq)
        xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
        xseq = xseq.reshape(bs, self.joint_num, nframes, -1)
        xseq = xseq.view(bs, 32, -1)

        
        for block in self.cross_attn_blocks:
            xseq = block(xseq, at_feat)

        xseq = xseq.view(bs, njoints, 32, -1)
        for block in self.mytimmblocks:
            xseq = block(xseq)
        
        output = xseq                

        output = self.output_process(output)
        return output[...,:noise_length]


    @staticmethod
    def apply_rotary(x, sinusoidal_pos):
        sin, cos = sinusoidal_pos
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).flatten(-2, -1)