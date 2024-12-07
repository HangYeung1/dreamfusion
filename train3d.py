import torch
import matplotlib.pyplot as plt
import numpy as np
from nerf.nerf import NeRF
import torch.nn as nn
import matplotlib.animation as animation
from guidance import StableGuide

from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


focal = 138
height, width = 512, 512


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
            torch.Tensor(
                [
                    [-9.9990219e-01, 4.1922452e-03, -1.3345719e-02, -5.3798322e-02],
                    [-1.3988681e-02, -2.9965907e-01, 9.5394367e-01, 3.8454704e00],
                    [-4.6566129e-10, 9.5403719e-01, 2.9968831e-01, 1.2080823e00],
                    [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
                ]
            ).to(device),
        )
        rays_o, rays_d = model.module.define_rays(height, width, focal, pose)
        rgb, _ = model.module.render(model.module, rays_o, rays_d, near, far, n_samples)
        rendered_images.append(rgb.cpu().numpy())
    return rendered_images


def visualize(checkpoint_path):
    nerf = NeRF().to(device)
    nerf = nn.DataParallel(nerf).to(device)
    ckpt = torch.load(f"{checkpoint_path}/ckpt.pth")
    nerf.module.load_state_dict(ckpt)
    height, width = 100, 100
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


def random_camera_pose_with_rotation(radius=1.0):
    theta = torch.acos(2 * torch.rand(1) - 1)
    phi = 2 * torch.pi * torch.rand(1)

    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    position = torch.stack([x, y, z]).view(3, 1)

    yaw = 2 * torch.pi * torch.rand(1)
    pitch = torch.acos(2 * torch.rand(1) - 1) - torch.pi / 2
    roll = 2 * torch.pi * torch.rand(1)

    R_yaw = torch.tensor(
        [
            [torch.cos(yaw), -torch.sin(yaw), 0],
            [torch.sin(yaw), torch.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    R_pitch = torch.tensor(
        [
            [torch.cos(pitch), 0, torch.sin(pitch)],
            [0, 1, 0],
            [-torch.sin(pitch), 0, torch.cos(pitch)],
        ]
    )
    R_roll = torch.tensor(
        [
            [1, 0, 0],
            [0, torch.cos(roll), -torch.sin(roll)],
            [0, torch.sin(roll), torch.cos(roll)],
        ]
    )
    rotation_matrix = R_yaw @ R_pitch @ R_roll

    transformation_matrix = torch.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position.view(3)

    return transformation_matrix


def train(checkpoint_path, n_iters=1000):
    model = NeRF().to(device)
    model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=5e-3, eps=1e-7)

    sd = StableGuide(
        t_range=(0.02, 0.98), guidance_scale=15, device=device, dtype=torch.float16
    )

    pos_embeds = sd.encode_text("An apple")
    neg_embeds = sd.encode_text("")

    plot_step = 500
    n_samples = 64

    for i in tqdm(range(n_iters + 1)):
        pose = random_camera_pose_with_rotation().to(device)

        rays_o, rays_d = model.module.define_rays(height, width, focal, pose)
        rgb, _ = model.module.render(
            model.module,
            rays_o,
            rays_d,
            near=2.0,
            far=6.0,
            n_samples=n_samples,
            rand=True,
        )

        optimizer.zero_grad()
        loss = sd.calculate_sds_loss(sd.encode_images(rgb), pos_embeds, neg_embeds)
        loss.backward()
        optimizer.step()

        if i % plot_step == 0:
            torch.save(model.module.state_dict(), f"{checkpoint_path}/ckpt.pth")
            with torch.no_grad():
                rays_o, rays_d = model.module.define_rays(height, width, focal, pose)
                rgb, depth = model.module.render(
                    model.module,
                    rays_o,
                    rays_d,
                    near=2.0,
                    far=6.0,
                    n_samples=n_samples,
                )

                plt.figure(figsize=(9, 3))

                plt.subplot(131)
                picture = rgb.cpu()
                plt.imshow(picture)
                plt.title(f"RGB Iter {i}")

                plt.subplot(132)
                picture = depth.cpu() * (rgb.cpu().mean(-1) > 1e-2)
                plt.imshow(picture, cmap="gray")
                plt.title(f"Depth Iter {i}")

                plt.show()


checkpoint = "output"
train(checkpoint)
# if not os.path.exists(f"{checkpoint}/ckpt.pth"):
#     train(checkpoint)
# visualize(checkpoint)
