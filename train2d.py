from parser import Config

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def train2d(args: Config):
    """Train loop for 2D SDS.

    Arguments:
        args: Data class object with training configuration.
    """

    # Initialize guidance
    guide = args.guide(
        t_range=args.t_range,
        guidance_scale=args.guidance_scale,
        device=args.device,
        dtype=args.dtype,
    )

    training = nn.Parameter(
        torch.rand(
            args.batch_size, *guide.train_shape, dtype=args.dtype, device=args.device
        ),
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [training], lr=args.lr, eps=1e-4, weight_decay=args.weight_decay
    )
    pos_embeds = guide.encode_text(args.prompt)
    neg_embeds = guide.encode_text(args.negative_prompt)

    # Optimize training tensor
    for _ in tqdm(range(args.iterations), desc="Optimizing images...:"):
        optimizer.zero_grad()
        loss = guide.calculate_sds_loss(training, pos_embeds, neg_embeds)
        loss.backward()
        optimizer.step()

    # Save images
    args.output_path.mkdir(parents=True, exist_ok=True)
    images = guide.decode_train(training)
    pil_images = [to_pil_image(image) for image in images]
    for idx, image in enumerate(pil_images):
        image.save(args.output_path / f"./image_{idx}.png")
