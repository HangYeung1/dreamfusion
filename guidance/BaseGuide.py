from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
from diffusers import SchedulerMixin
from transformers import PreTrainedModel, PreTrainedTokenizer


class BaseGuide(nn.Module, ABC):
    """Diffusion guidance wrapper base class.

    Attributes:
        min_step (int): Minimum diffusion timestep.
        max_step (int): Maximum diffusion timestep.
        train_shape (Tuple[int, int, int]): Training tensor shape.
        guidance_scale (float): Classifier-free guidance weight.
        tokenizer (PreTrainedTokenizer): Diffuser's Tokenizer.
        text_encoder (PreTrainedModel): Diffuser's text encoder.
        scheduler (SchedulerMixin): Diffuser's scheduler.
        device (torch.device): Device of guidance.
        dtype (torch.dtype): Precision of guidance.
    """

    def __init__(
        self,
        *,
        t_range: Tuple[float, float],
        guidance_scale: float,
        train_shape: Tuple[int, int, int],
        tokenizer: PreTrainedTokenizer,
        text_encoder: PreTrainedModel,
        scheduler: SchedulerMixin,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize guidance components.

        Arguments:
            t_range: Diffusion time interval.
            guidance_scale: Classifier-free guidance weight.
            train_shape: Training tensor shape.
            tokenizer: Diffuser's Tokenizer.
            text_encoder: Diffuser's text encoder.
            scheduler: Diffuser's scheduler.
            device: Device of guidance.
            dtype: Precision of guidance.
        """
        super().__init__()

        # Set basic members
        self.device = device
        self.dtype = dtype

        self.guidance_scale = guidance_scale
        self.train_shape = train_shape

        # Set diffusion components
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

        num_train_steps = self.scheduler.config.num_train_timesteps
        step_range = [int(t * num_train_steps) for t in t_range]
        self.min_step, self.max_step = step_range

    def calculate_sds_loss(
        self, training: torch.Tensor, pos_embeds: torch.Tensor, neg_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Calculate score distillation sampling loss.

        Arguments:
            training: Training tensor of shape (N, *self.train_shape).
            pos_embeds: Target text embedding tensor.
            neg_embeds: Avoidance text embedding tensor.

        Returns:
            SDS loss tensor of shape (N,).
        """
        N, _, _, _ = training.shape

        # Add noise
        timesteps = torch.randint(
            self.min_step, self.max_step + 1, (N,), device=self.device
        )
        noise = torch.randn_like(training, device=self.device)
        train_noisy = self.scheduler.add_noise(training, noise, timesteps)

        # Calculate loss via reparameterization
        noise_pred = self.predict_cfg_noise(
            train_noisy, timesteps, pos_embeds, neg_embeds
        )
        weight = 1 - self.alphas[timesteps].view(-1, 1, 1, 1).expand(
            -1, *self.train_shape
        )
        gradient = weight * (noise_pred - noise)
        sds_loss = (gradient * training).sum()

        return sds_loss

    @abstractmethod
    def predict_noise(
        self,
        train_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Abstract. Get noise prediction.

        Arguments:
            training: Training tensor of shape (N, *self.train_shape).
            timesteps: Timestep tensor of shape (N,).
            embeds: Text embedding tensor.

        Returns:
            Noise prediction tensor of shape (N, *self.train_shape).
        """
        pass

    @torch.no_grad
    def predict_cfg_noise(
        self,
        train_noisy: torch.Tensor,
        timesteps: torch.IntTensor,
        pos_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate classifier-free noise.

        Arguments:
            train_noisy: Noisy training tensor of shape (N, *self.train_shape).
            timesteps: Timestep tensor of shape (N,).
            pos_embeds: Target text embedding tensor.
            neg_embeds: Avoidance text embedding tensor.

        Returns:
            Predicted noise tensor of shape (N, *self.train_shape).
        """
        N, _, _, _ = train_noisy.shape

        # Predict positive and negative noise
        train_comb = torch.cat([train_noisy] * 2, dim=0)
        timesteps_comb = torch.cat([timesteps] * 2, dim=0)
        embeds_comb = torch.cat(
            [
                torch.cat([pos_embeds] * N, dim=0),
                torch.cat([neg_embeds] * N, dim=0),
            ],
            dim=0,
        )

        noise_pred_comb = self.predict_noise(train_comb, timesteps_comb, embeds_comb)
        noise_pred_pos, noise_pred_neg = noise_pred_comb.chunk(2)

        # Calculate classifier-free noise
        noise_pred = noise_pred_neg + self.guidance_scale * (
            noise_pred_pos - noise_pred_neg
        )

        return noise_pred

    @torch.no_grad
    def encode_text(self, text: str) -> torch.Tensor:
        """Embed text.

        Arguments:
            text: Target text.

        Returns:
            Text embedding tensor.
        """
        input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.to(self.device)
        text_embeds = self.text_encoder(input_ids)[0]
        return text_embeds

    @abstractmethod
    def decode_train(self, training: torch.Tensor) -> torch.Tensor:
        """Abstract. Convert training tensor to images tensor.

        Arguments:
            training: Training tensor of shape (N, *train_shape)

        Returns:
            Images tensor.
        """
        pass
