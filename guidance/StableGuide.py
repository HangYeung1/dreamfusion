from typing import Tuple

import torch
from diffusers import DiffusionPipeline

from .BaseGuide import BaseGuide


class StableGuide(BaseGuide):
    """Stable Diffusion v2-1 guidance wrapper.

    Attributes:
        min_step (int): Minimum diffusion timestep.
        max_step (int): Maximum diffusion timestep.
        train_shape (Tuple[int, int, int]): Training tensor shape.
        guidance_scale (float): Classifier-free guidance weight.
        tokenizer (CLIPTokenizer): SD's Tokenizer.
        text_encoder (CLIPTextModel): SD's text encoder.
        unet (UNet2DConditionModel): SD's UNet.
        vae (AutoencoderKL): SD's variational autoencoder.
        scheduler (SchedulerMixin): SD's scheduler.
        device (torch.device): Device of guidance.
        dtype (torch.dtype): Precision of guidance.
    """

    def __init__(
        self,
        t_range: Tuple[float, float],
        guidance_scale: float,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize guidance components from Stable Diffusion.

        Arguments:
            t_range: Diffusion time interval.
            guidance_scale: Classifier-free guidance weight.
            device: Device of guidance.
            dtype: Precision of guidance.
        """
        sd = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base", torch_dtype=dtype
        ).to(device)

        super().__init__(
            t_range=t_range,
            guidance_scale=guidance_scale,
            train_shape=(4, 64, 64),
            tokenizer=sd.tokenizer,
            text_encoder=sd.text_encoder,
            scheduler=sd.scheduler,
            device=device,
            dtype=dtype,
        )

        self.unet = sd.unet
        self.vae = sd.vae

    @torch.no_grad
    def predict_noise(
        self,
        train_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Get noise prediction.

        Arguments:
            training: Training tensor of shape (N, *self.train_shape).
            timesteps: Timestep tensor of shape (N,).
            embeds: Text embedding tensor.

        Returns:
            Noise prediction tensor of shape (N, *self.train_shape).
        """
        return self.unet(train_noisy, timesteps, encoder_hidden_states=embeds).sample

    @torch.no_grad
    def decode_train(self, training: torch.Tensor) -> torch.Tensor:
        """Convert training tensor to images tensor.

        Arguments:
            training: Training tensor of shape (N, *train_shape).

        Returns:
            Images tensor.
        """
        train_scaled = training / self.vae.config.scaling_factor
        vae_out = self.vae.decode(train_scaled).sample
        images = (vae_out / 2 + 0.5).clamp(0, 1)
        return images

    @torch.no_grad
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Convert images tensor to training tensor.

        Arguments:
            images: Images tensor.

        Returns:
            Training tensor of shape (N, *train_shape).
        """
        images_scaled = images * 2 - 1
        vae_out = self.vae.encode(images_scaled).latent_dist.sample()
        training = vae_out * self.vae.config.scaling_factor
        return training
