import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers import DDIMScheduler, StableDiffusionPipeline

import logging

class Stable(nn.Module):
    """
    Wrapper for Stable Diffusion
    """

    def __init__(self, device, dtype):
       """Init the Image Diffusion model"""
       super().__init__()
       self.device = device
       self.dtype = dtype
       self.min_time_step = 0.2
       self.max_time_step = 0.8

       logging.info("Loading Stable Diffusion...")

       key = "Whatever"

       self.model = StableDiffusionPipeline.from_pretrained(key, torch_dtype=self.dtype)
       self.model.to(self.device)

       self.tokenizer = self.model.tokenizer
       self.unet = self.model.unet
       self.scheduler = DDIMScheduler(
           model=self.model,
           subfolder="whatever",
           dtype=self.dtype
       )
       logging.info("Stable Diffusion Loaded")
    
    def text_embed(self, prompt, neg=""):
        text_emb, uncond = self.model.encode_prompt(prompt, negative_prompt=neg, device=self.device)
        text_emb = torch.cat([uncond, text_emb], dim=1)
        return text_emb
    
    # [TODO] implement/fix the function
    def sds_loss(self, text_emb, guidance_scale=7.5, grad_scale=1.0):
        """Compute the Score Distillation Sampling (SDS) loss."""
        # Random timestep selection within the range
        #[TODO] need to check the *1000
        #
        timesteps = torch.randint(int(self.min_time_step * 1000), int(self.max_time_step * 1000), (1,), device=self.device)

        # Generate a noise tensor with the same shape as the U-Net's input
        noise = torch.randn_like(text_emb, device=self.device)

        # Add noise to the latent representation
        noisy_latents = text_emb + noise * self.scheduler.get_alpha(timesteps).sqrt()

        # Calculate the score prediction
        with torch.no_grad():
            predicted_noise = self.unet(noisy_latents, timesteps, encoder_hidden_states=text_emb).sample

        # Compute the SDS loss
        sds_loss = F.mse_loss(predicted_noise, noise) * grad_scale
        return sds_loss


    # [NOTE] Encode/decode may need scaling factors
    @torch.no_grad
    def latent_to_img(self, latents):
        return self.model.decode_latents(latents).images[0]

    @torch.no_grad
    def img_to_latent(self, img):
        return self.model.vae.encode(img).latent_dist.sample()