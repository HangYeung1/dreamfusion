import cv2
import numpy as np
import torch

from nerf.nerf import NeRF

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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


input_path = "output/apple0/ckpt9500.pth"
output_path = "output/apple0/9500.mp4"
height = width = 512
focal = 450
n_samples = 128
frame_rate = 10
render_interval = 5
near = 2.0
far = 10.0

fourcc = cv2.VideoWriter_fourcc(*"H264")
out = cv2.VideoWriter(output_path, fourcc, 10, (height, width))

nerf = NeRF().to(device)
ckpt = torch.load(input_path)
nerf.load_state_dict(ckpt)
with torch.no_grad():
    for i in range(0, 360, render_interval):
        theta = torch.tensor(i)
        phi = torch.tensor(75)
        pose = get_camera_pose(theta, phi, 3.5).to(device)

        rays_o, rays_d = nerf.define_rays(height, width, focal, pose)
        rgb, depth = nerf.render(
            nerf,
            rays_o,
            rays_d,
            near=near,
            far=far,
            n_samples=n_samples,
        )

        out.write((rgb * 255).cpu().numpy().astype(np.uint8))

out.release()
