import logging
from typing import Tuple

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline


class StableGuide(nn.Module):
    """A Stable Diffusion v2-1 (SD) Guidance Wrapper.

    This class contains a disassembled SD pipeline for Score Distillation
    Sampling (SDS). Note that SD is latent space!

    Attributes:
        device (torch.device): Device where guidance lives.
        dtype (torch.dtype): Precision of guidance.
        min_step (int): Minimum diffusion timestep.
        max_step (int): Maximum diffusion timestep.
        guidance_scale (float): Classifier-free guidance weight.
        tokenizer (CLIPTokenizer): Tokenizer from SD.
        text_encoder (CLIPTextModel): Text encoder from SD.
        unet (UNet2DConditionModel): UNet from SD.
        vae (AutoencoderKL): Variational autoencoder from SD.
        scheduler (SchedulerMixin): Scheduler for diffusion control.
    """

    def __init__(
        self,
        t_range: Tuple[float, float],
        guidance_scale: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize guidance components from StableDiffusionPipeline.

        Args:
            t_range: Diffusion sampling interval in [0, 1].
            guidance_scale: Classifier-free guidance weight.
            device: Device where guidance lives.
            dtype: Precision of guidance.
        """
        super().__init__()

        # Set hardware
        self.device = device
        self.dtype = dtype

        # Extract components from SD
        logging.info("Loading Stable Diffusion pipeline...")

        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", torch_dtype=self.dtype
        ).to(self.device)

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

        self.unet = pipe.unet
        self.vae = pipe.vae

        # NOTE: May need to change to DDIM
        self.scheduler = pipe.scheduler
        self.betas = self.scheduler.betas.to(self.device)

        logging.info("Stable Diffusion loaded!")

        # Set guidance parameters
        num_train_steps = self.scheduler.config.num_train_timesteps
        step_range = [int(t * num_train_steps) for t in t_range]
        self.min_step, self.max_step = step_range

        self.guidance_scale = guidance_scale

    def calculate_sds_loss(
        self, latents: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Calculate SDS loss.

        Args:
            latents: Latent tensor of shape (N, 4, 64, 64).
            text_embeds: Text embed tensor of shape ?

        Returns:
            SDS loss tensor of shape ?
        """
        N, _, _, _ = latents.shape

        # Add noise to latent
        timesteps = torch.randint(
            self.min_step, self.max_step + 1, (N,), device=self.device
        )
        noise = torch.randn_like(latents, device=self.device)
        latents_noisy = self.scheduler.add_noise(latents, noise, timesteps)

        # Calculate loss from predicted noise
        # Weighing factor w(t) = variance_t = beta_t
        # Reparameterization trick from DreamFusion paper
        noise_pred = self.predict_noise(latents_noisy, timesteps, text_embeds)
        weight_t = self.betas[timesteps]
        sds_loss = (weight_t * noise_pred * latents).sum()

        return sds_loss

    def init_latents(self, batch_size: int) -> torch.Tensor:
        """Create new latents.

        Args:
            batch_size: Number of requested latents.

        Returns:
            New latent tensor of shape (batch_size, 4, 64, 64).
        """
        return nn.Parameter(
            torch.rand(
                batch_size, 4, 64, 64, dtype=self.dtype, device=self.device
            ),
            requires_grad=True,
        )

    @torch.no_grad
    def predict_noise(
        self,
        latents_noisy: torch.Tensor,
        timesteps: torch.IntTensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise with classifier-free guidance.

        Args:
            latents_noisy: Noisy latent tensor of shape (N, 4, 64, 64).
            timesteps: Timestep tensor of shape (N,).
            text_embeds: Text embed tensor of shape ?

        Returns:
            Predicted noise tensor of shape (N, 4, 64, 64).
        """

        # Get unconditioned and conditioned noise
        latents_dup = torch.cat([latents_noisy, latents_noisy], dim=0)
        timesteps_dup = torch.cat([timesteps, timesteps], dim=0)
        text_embeds_dup = torch.cat(
            [torch.zeros_like(text_embeds), text_embeds], dim=0
        )

        noise_pred_combined = self.unet(
            latents_dup, timesteps_dup, encoder_hidden_states=text_embeds_dup
        ).sample
        noise_pred_uncond, noise_pred_cond = noise_pred_combined.chunk(2)

        # Classifier-free weighing
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )
        return noise_pred

    @torch.no_grad
    def encode_text(self, prompt: str) -> torch.Tensor:
        """Embed text for guidance.

        Args:
            prompt: Guiding prompt.

        Returns:
            Text embed tensor of shape ?
        """
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.device)
        text_embeds = self.text_encoder(input_ids)[0]
        return text_embeds

    @torch.no_grad
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Convert latents to images.

        Args:
            latents: Latent tensor of shape (N, 4, 64, 64).

        Returns:
            Image tensor of shape (N, 3, 512, 512).
        """
        latents_scaled = latents / self.vae.config.scaling_factor
        vae_out = self.vae.decode(latents_scaled).sample
        imgs = ((vae_out + 1) / 2).clamp(0, 1)  # Clamp for numerical stability
        return imgs

    @torch.no_grad
    def encode_imgs(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to latents.

        Args:
            imgs: Image tensor of shape (N, 3, 512, 512)

        Returns:
            Latent tensor of shape (N, 4, 64, 64)
        """
        imgs_scaled = imgs * 2 - 1
        vae_out = self.vae.encode(imgs_scaled).latent_dist.sample()
        latents = vae_out * self.vae.config.scaling_factor
        return latents
