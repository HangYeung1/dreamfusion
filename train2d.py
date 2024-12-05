from parser import Config

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

import guidance


def get_init(args: Config):
    """Get initial 2d tensors."""
    shape = None
    match args.guide:
        case guidance.StableGuide:
            shape = (args.batch_size, 4, 64, 64)
        case guidance.IFGuide:
            shape = (args.batch_size, 3, 64, 64)

    return nn.Parameter(
        torch.rand(*shape, dtype=args.dtype, device=args.device),
        requires_grad=True,
    )


def get_images(trained: torch.Tensor, model: guidance.GuideTypes):
    """Convert trained tensors to PIL images."""
    images = None
    match type(model):
        case guidance.StableGuide:
            images = model.decode_latents(trained)
        case _:
            images = trained

    return [to_pil_image(image) for image in images]


def train2d(args: Config):
    """Train loop for 2D SDS.

    Args:
        args: Data class object with training configuration.
    """

    args.output_path.mkdir(parents=True, exist_ok=True)

    # Optimize images
    model = args.guide(
        t_range=args.t_range,
        guidance_scale=args.guidance_scale,
        device=args.device,
        dtype=args.dtype,
    )

    latents = get_init(args)
    optimizer = torch.optim.Adam([latents], lr=args.lr, eps=1e-4)
    text_embeds = model.encode_text(args.prompt)

    for _ in tqdm(range(args.iterations), desc="Optimizing images...:"):
        optimizer.zero_grad()
        loss = model.calculate_sds_loss(latents, text_embeds)
        loss.backward()
        optimizer.step()

    # Save images
    images = get_images(latents, model)
    for idx, image in enumerate(images):
        image.save(args.output_path / f"./image_{idx}.png")
