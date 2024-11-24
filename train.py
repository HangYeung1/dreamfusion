from parser import SDSConfig

import torch
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm


def train(args: SDSConfig):
    """Train loop for 2D SDS.

    Args:
        args: Data class object with training configuration.
    """

    model = args.guide(
        t_range=args.t_range,
        guidance_scale=args.guidance_scale,
        device=args.device,
        dtype=args.dtype,
    )

    # Optimize image
    latents = model.init_latents(1)
    optimizer = torch.optim.Adam([latents], lr=args.lr)

    text_embeds = model.encode_text(args.prompt)

    for _ in tqdm(range(args.iterations)):
        optimizer.zero_grad()
        loss = model.calculate_sds_loss(latents, text_embeds)
        loss.backward()
        optimizer.step()

    # Save image
    image = model.decode_latents(latents)[0]
    to_pil_image(image).save("./test.png")
