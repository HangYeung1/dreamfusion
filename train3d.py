import torch
import matplotlib.pyplot as plt
import numpy as np
from nerf.nerf import NeRF
import torch.nn as nn
import matplotlib.animation as animation

from tqdm import tqdm
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

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


def render_rotating_nerf(
    model, height, width, focal, poses, n_samples, near=2.0, far=6.0, n_frames=120
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
        pose = torch.matmul(rotation_matrix, poses[0])
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
        rotating_images = render_rotating_nerf(
            nerf, height, width, focal, poses, n_samples=64
        )

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


def train(checkpoint_path, n_iters=3000):
    model = NeRF().to(device)
    model = nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.module.parameters(), lr=5e-3, eps=1e-7)

    plot_step = 500
    n_samples = 64

    for i in tqdm(range(n_iters + 1)):
        images_idx = np.random.randint(images.shape[0])
        target = images[images_idx]
        pose = poses[images_idx]
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
        image_loss = torch.nn.functional.mse_loss(rgb, target)
        image_loss.backward()
        optimizer.step()

        if i % plot_step == 0:
            torch.save(model.module.state_dict(), f"{checkpoint_path}/ckpt.pth")
            with torch.no_grad():
                rays_o, rays_d = model.module.define_rays(
                    height, width, focal, testpose
                )
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
if not os.path.exists(f"{checkpoint}/ckpt.pth"):
    train(checkpoint)
visualize(checkpoint)
