import logging
from typing import Tuple

import torch
import torch.nn as nn
from diffusers import DiffusionPipeline


class IFGuide(nn.Module):
    """DeepFloyd IF Guidance Wrapper.

    Attributes:
        device (torch.device): Device where guidance lives.
        dtype (torch.dtype): Precision of guidance.
        min_step (int): Minimum diffusion timestep.
        max_step (int): Maximum diffusion timestep.
        guidance_scale (float): Classifier-free guidance weight.
        tokenizer (T5Tokenizer): Tokenizer from IF.
        text_encoder (T5EncoderModel): Text encoder from IF.
        unet (UNet2DConditionModel): UNet from IF.
        scheduler (DDPMScheduler): Scheduler for diffusion control.
    """

    def __init__(
        self,
        t_range: Tuple[float, float],
        guidance_scale: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize guidance components from DiffusionPipeline.

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

        # Extract components from IF
        logging.info("Loading IF pipeline...")

        pipe = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0", torch_dtype=self.dtype
        ).to(self.device)

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.betas = self.scheduler.betas.to(self.device)

        logging.info("IF loaded!")

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
            latents: Latent tensor of shape (N, 3, 64, 64).
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
        weight_t = self.betas[timesteps].view(-1, 1, 1, 1).expand(-1, 3, 64, 64)
        sds_loss = (weight_t * noise_pred * latents).sum()

        return sds_loss

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
        N, _, _, _ = latents_noisy.shape

        # Get unconditioned and conditioned noise
        latents_dup = torch.cat([latents_noisy] * 2, dim=0)
        timesteps_dup = torch.cat([timesteps] * 2, dim=0)
        text_embeds_dup = torch.cat(
            [
                torch.cat([torch.zeros_like(text_embeds)] * N, dim=0),
                torch.cat([text_embeds] * N, dim=0),
            ],
            dim=0,
        )

        noise_pred_combined = self.unet(
            latents_dup, timesteps_dup, encoder_hidden_states=text_embeds_dup
        ).sample[:, :3]
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
