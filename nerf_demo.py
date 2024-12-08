import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from nerf.nerf import NeRF

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

rawData = np.load("tiny_nerf_data.npz", allow_pickle=True)
images = rawData["images"]
poses = rawData["poses"]
focal = rawData["focal"]
height, width = images.shape[1:3]
height = int(height)
width = int(width)

testimg, testpose = images[99], poses[99]
images = torch.Tensor(images).to(device)
poses = torch.Tensor(poses).to(device)
testimg = torch.Tensor(testimg).to(device)
testpose = torch.Tensor(testpose).to(device)


def train(checkpoint_path, n_iters=3000):
    model = NeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, eps=1e-4)

    plot_step = 500
    n_samples = 64

    for i in tqdm(range(n_iters)):
        images_idx = np.random.randint(images.shape[0])
        target = images[images_idx]
        pose = poses[images_idx]
        rays_o, rays_d = model.define_rays(height, width, focal, pose)
        rgb, _ = model.render(
            model,
            rays_o,
            rays_d,
            near=2.0,
            far=6.0,
            n_samples=n_samples,
            rand=True,
        )

        optimizer.zero_grad()
        image_loss = torch.nn.functional.mse_loss(rgb, target)
        image_loss.backward()
        optimizer.step()

        if i % plot_step == 0:
            torch.save(model.state_dict(), f"{checkpoint_path}/ckpt{i}.pth")
            with torch.no_grad():
                rays_o, rays_d = model.define_rays(height, width, focal, testpose)
                rgb, depth = model.render(
                    model,
                    rays_o,
                    rays_d,
                    near=2.0,
                    far=6.0,
                    n_samples=n_samples,
                )

                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(rgb.cpu())
                plt.subplot(1, 2, 2)
                plt.imshow(depth.cpu() * (rgb.cpu().mean(-1) > 1e-2), cmap="gray")
                plt.show()
        torch.save(model.state_dict(), f"{checkpoint_path}/final.pth")


@torch.no_grad
def get_camera_pose(theta_deg, phi_deg, radius):
    """Math style. Theta is azimuth. Phi is incline."""
    theta_rad = torch.deg2rad(torch.tensor(theta_deg))
    phi_rad = torch.deg2rad(torch.tensor(phi_deg))

    x = radius * torch.sin(phi_rad) * torch.cos(theta_rad)
    y = radius * torch.sin(phi_rad) * torch.sin(theta_rad)
    z = radius * torch.cos(phi_rad)
    position = torch.tensor([x, y, z])

    # Get direction vectors
    up = torch.tensor([0.0, 0.0, 1.0])

    forward = position
    forward = forward / torch.norm(forward)

    right = torch.cross(up, forward)
    right = right / torch.norm(right)

    corrected_up = torch.cross(forward, right)

    rotation = torch.stack([right, corrected_up, forward], dim=1)

    pose = torch.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = position

    return pose


checkpoint_path = "output"

# train(checkpoint_path)
plt.ion()
nerf = NeRF().to(device)
ckpt = torch.load(f"{checkpoint_path}/ckpt3250.pth")
nerf.load_state_dict(ckpt)
with torch.no_grad():
    while True:
        for i in range(0, 360, 10):
            theta = i
            phi = 70
            pose = get_camera_pose(theta, phi, 3.5).to(device)

            rays_o, rays_d = nerf.define_rays(height, width, 135, pose)
            rgb, depth = nerf.render(
                nerf,
                rays_o,
                rays_d,
                near=2.0,
                far=6.0,
                n_samples=64,
            )

            plt.imshow(rgb.cpu())
            plt.draw()
            plt.pause(0.1)
