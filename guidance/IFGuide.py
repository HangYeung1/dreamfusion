from typing import Tuple

import torch
from diffusers import DiffusionPipeline

from .BaseGuide import BaseGuide


class IFGuide(BaseGuide):
    """DeepFloyd IF guidance wrapper.

    Attributes:
        min_step (int): Minimum diffusion timestep.
        max_step (int): Maximum diffusion timestep.
        train_shape (Tuple[int, int, int]): Training tensor shape.
        guidance_scale (float): Classifier-free guidance weight.
        tokenizer (CLIPTokenizer): IF's Tokenizer.
        text_encoder (CLIPTextModel): IF's text encoder.
        unet (UNet2DConditionModel): IF's UNet.
        vae (AutoencoderKL): IF's variational autoencoder.
        scheduler (SchedulerMixin): IF's scheduler.
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
        """Initialize guidance components from DeepFloyd IF.

        Arguments:
            t_range: Diffusion time interval.
            guidance_scale: Classifier-free guidance weight.
            device: Device of guidance.
            dtype: Precision of guidance.
        """
        if_ = DiffusionPipeline.from_pretrained(
            "DeepFloyd/IF-I-L-v1.0", torch_dtype=dtype
        ).to(device)

        super().__init__(
            t_range=t_range,
            guidance_scale=guidance_scale,
            train_shape=(3, 64, 64),
            tokenizer=if_.tokenizer,
            text_encoder=if_.text_encoder,
            scheduler=if_.scheduler,
            device=device,
            dtype=dtype,
        )

        self.unet = if_.unet

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
        return self.unet(train_noisy, timesteps, encoder_hidden_states=embeds).sample[
            :, :3
        ]

    @torch.no_grad
    def decode_train(self, training: torch.Tensor) -> torch.Tensor:
        """Convert training tensor to images tensor.

        Arguments:
            training: Training tensor of shape (N, *train_shape).

        Returns:
            Images tensor.
        """
        return training / 2 + 0.5
