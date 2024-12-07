import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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

focal = 135
height, width = 64, 64


def render_rotating_nerf(
    model, height, width, focal, n_samples, near=2.0, far=6.0, n_frames=120
):
    rendered_images = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        rotation_matrix = torch.tensor(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float32,
        ).to(device)
        pose = torch.matmul(
            rotation_matrix,
            get_camera_pose(torch.tensor(0), torch.tensor(0), 4).to(device),
        )
        rays_o, rays_d = model.module.define_rays(height, width, focal, pose)
        rgb, _ = model.module.render(model.module, rays_o, rays_d, near, far, n_samples)
        rendered_images.append(rgb.cpu().numpy())
    return rendered_images


def visualize(checkpoint_path):
    nerf = NeRF().to(device)
    nerf = nn.DataParallel(nerf).to(device)
    ckpt = torch.load(f"{checkpoint_path}/ckpt_final.pth")
    nerf.module.load_state_dict(ckpt)
    height, width = 64, 64
    with torch.no_grad():
        rotating_images = render_rotating_nerf(nerf, height, width, focal, n_samples=64)

        fig = plt.figure()
        im = plt.imshow(rotating_images[0])

        def update(frame):
            im.set_array(rotating_images[frame])
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=len(rotating_images), interval=1000 / 30
        )
        ani.save(f"{checkpoint_path}/rotating_video.mp4", writer="ffmpeg", fps=30)
        plt.show()


@torch.no_grad
def get_camera_pose(theta, phi, r=1.0):
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)

    # Normalize the forward direction vector
    forward = torch.tensor([x, y, z], dtype=torch.float32)
    forward = forward / torch.linalg.norm(forward)

    # Default right and up vectors
    right = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    up = torch.linalg.cross(right, forward)

    # Recompute right to ensure orthogonality
    right = torch.linalg.cross(up, forward)

    # Normalize right and up
    right = right / torch.linalg.norm(right)
    up = up / torch.linalg.norm(up)

    # Build the 4x4 camera matrix
    camera_matrix = torch.eye(4, dtype=torch.float32)
    camera_matrix[0, :3] = right
    camera_matrix[1, :3] = up
    camera_matrix[2, :3] = -forward

    # Translation: Position of the camera in space
    camera_matrix[0, 3] = -torch.dot(
        right, torch.tensor([x, y, z], dtype=torch.float32)
    )
    camera_matrix[1, 3] = -torch.dot(up, torch.tensor([x, y, z], dtype=torch.float32))
    camera_matrix[2, 3] = torch.dot(
        forward, torch.tensor([x, y, z], dtype=torch.float32)
    )

    return camera_matrix


def train(checkpoint_path, n_iters=3000):
    model = NeRF().to(device)
    model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(
        model.module.parameters(), lr=0.001, eps=1e-4, weight_decay=0
    )

    guide = IFGuide(
        t_range=(0.02, 0.98), guidance_scale=10, device=device, dtype=torch.float16
    )

    pos_embeds = guide.encode_text("A DSLR photo of an apple")
    neg_embeds = guide.encode_text("")

    plot_step = 250
    n_samples = 64

    for i in tqdm(range(n_iters)):
        theta = torch.rand(1) * 2 * torch.pi
        phi = torch.rand(1) * torch.pi
        pose = get_camera_pose(theta, phi, 4).to(device)

        rays_o, rays_d = model.module.define_rays(height, width, focal, pose)
        rgb = (
            model.module.render(
                model.module,
                rays_o,
                rays_d,
                near=2,
                far=6,
                n_samples=n_samples,
                rand=True,
            )[0]
            .permute(1, 2, 0)
            .view(1, 3, 64, 64)
            .half()
        )

        optimizer.zero_grad()
        loss = guide.calculate_sds_loss(rgb, pos_embeds, neg_embeds)
        loss.backward()
        optimizer.step()

        if i % plot_step == 0:
            torch.save(model.module.state_dict(), f"{checkpoint_path}/ckpt{i}.pth")

            with torch.no_grad():
                rays_o, rays_d = model.module.define_rays(
                    height,
                    width,
                    focal,
                    get_camera_pose(torch.tensor(0), torch.tensor(0), 4).to(device),
                )
                test, _ = model.module.render(
                    model.module,
                    rays_o,
                    rays_d,
                    near=2,
                    far=6,
                    n_samples=n_samples,
                    rand=True,
                )
                plt.figure()
                plt.imshow(test.detach().cpu())
                plt.show()

    torch.save(model.module.state_dict(), f"{checkpoint_path}/ckpt_final.pth")


checkpoint = "output"
train(checkpoint)
# visualize(checkpoint)
