from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from guidance import IFGuide
from nerf.nerf import NeRF

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


torch.manual_seed(42)
torch.cuda.manual_seed(42)

height = width = 64
prompt = "an apple"
negative_prompt = "cropped, out of frame, morbid, mutilated, bizarre, corrupted, malformed, low quality, artifacts, watermark, signature"
plot_step = 250
n_samples = 64
render_dist = 4.0
near = 2.0
far = 8.0
save_name = "apple4"
save_path = "output"

lr = 1e-4
eps = 1e-4
weight_decay = 0.1
guidance = 100

save_path = Path(save_path) / save_name
weights_path = save_path / "weights"
images_path = save_path / "images"

weights_path.mkdir(parents=True, exist_ok=True)
images_path.mkdir(parents=True, exist_ok=True)


@torch.no_grad
def get_camera_pose(theta_deg, phi_deg, radius):
    """Math style. Theta is azimuth. Phi is incline."""
    theta_rad = torch.deg2rad(theta_deg)
    phi_rad = torch.deg2rad(phi_deg)

    # Calculate position vector
    x = radius * torch.sin(phi_rad) * torch.cos(theta_rad)
    y = radius * torch.sin(phi_rad) * torch.sin(theta_rad)
    z = radius * torch.cos(phi_rad)
    position = torch.tensor([x, y, z])

    # Calculate rotation matrix
    up = torch.tensor([0.0, 0.0, 1.0])
    forward = position
    forward = forward / torch.norm(forward)
    right = torch.cross(up, forward)
    right = right / torch.norm(right)
    corrected_up = torch.cross(forward, right)
    rotation = torch.stack([right, corrected_up, forward], dim=1)

    # Compose intriniscs
    pose = torch.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position

    return pose


def train(n_iters=30000):
    model = NeRF().to(device)
    guide = IFGuide(
        t_range=(0.02, 0.98),
        guidance_scale=guidance,
        device=device,
        dtype=torch.float16,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay
    )

    neg_embeds = guide.encode_text(negative_prompt)
    pos_embeds_top = guide.encode_text("overhead view of " + prompt)
    pos_embeds_front = guide.encode_text("front view of " + prompt)
    pos_embeds_side = guide.encode_text("side view of " + prompt)
    pos_embeds_back = guide.encode_text("back view of " + prompt)

    for i in tqdm(range(n_iters)):
        theta = torch.rand(1) * 360
        phi = torch.rand(1) * 100
        pose = get_camera_pose(theta, phi, render_dist).to(device)
        focal = (torch.rand(1) * 0.65 + 0.7) * width

        rays_o, rays_d = model.define_rays(height, width, focal, pose)
        rgb = (
            model.render(
                model,
                rays_o,
                rays_d,
                near=near,
                far=far,
                n_samples=n_samples,
            )[0]
            .permute(2, 0, 1)
            .view(1, 3, height, width)
        ).half()

        # Get positionally modulated prompts
        curr_embeds = None
        if phi < 60:
            curr_embeds = pos_embeds_top
        elif theta <= 45 and theta > 315:
            curr_embeds = pos_embeds_front
        elif theta > 135 and theta < 225:
            curr_embeds = pos_embeds_back
        else:
            curr_embeds = pos_embeds_side

        optimizer.zero_grad()
        loss = guide.calculate_sds_loss(rgb, curr_embeds, neg_embeds)
        loss.backward()
        optimizer.step()

        if i % plot_step == 0:
            torch.save(model.state_dict(), weights_path / f"ckpt_{i}.pth")
            with torch.no_grad():
                rays_o, rays_d = model.define_rays(
                    height,
                    width,
                    width,
                    get_camera_pose(torch.tensor(0), torch.tensor(75), render_dist).to(
                        device
                    ),
                )
                test_rgb, test_depth = model.render(
                    model,
                    rays_o,
                    rays_d,
                    near=near,
                    far=far,
                    n_samples=n_samples,
                    rand=True,
                )

                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(test_rgb.cpu())
                plt.subplot(1, 2, 2)
                plt.imshow(
                    test_depth.cpu() * (test_rgb.cpu().mean(-1) > 1e-2), cmap="gray"
                )
                plt.savefig(images_path / f"pic_{i}.png")
                plt.close()

    torch.save(model.state_dict(), weights_path / "ckpt_final.pth")


train()
