import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.functional import resize
from tqdm import tqdm

from guidance import StableGuide
from nerf.nerf import NeRF

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


torch.manual_seed(42)
torch.cuda.manual_seed(42)

focal = 80
height = width = 128


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
        rays_o, rays_d = model.define_rays(height, width, focal, pose)
        rgb, _ = model.render(model, rays_o, rays_d, near, far, n_samples)
        rendered_images.append(rgb.cpu().numpy())
    return rendered_images


def visualize(checkpoint_path):
    nerf = NeRF().to(device)
    ckpt = torch.load(f"{checkpoint_path}/ckpt_final.pth")
    nerf.load_state_dict(ckpt)
    height, width = 128, 128
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
def get_camera_pose(theta_deg, phi_deg, radius, up=torch.tensor([0.0, 1.0, 0.0])):
    theta_rad = torch.deg2rad(theta_deg)
    phi_rad = torch.deg2rad(phi_deg)

    # Spherical to Cartesian conversion
    x = radius * torch.sin(phi_rad) * torch.cos(theta_rad)
    y = radius * torch.cos(phi_rad)  # Elevation
    z = radius * torch.sin(phi_rad) * torch.sin(theta_rad)
    position = torch.tensor([x, y, z])

    # Forward vector: from camera position toward the origin
    forward = -position / torch.norm(position)

    # If forward vector is close to the up vector, adjust the up vector
    if torch.abs(torch.dot(forward, up)) > 0.99:
        up = torch.tensor([1.0, 0.0, 0.0])

    # Right vector: perpendicular to up and forward
    right = torch.cross(up, forward)
    right = right / torch.norm(right)

    # Recompute the up vector to ensure orthogonality
    up = torch.cross(forward, right)

    # Construct the rotation matrix (camera-to-world transform)
    rotation = torch.stack([right, up, forward], dim=1)  # Camera's axes as columns

    # Create a 4x4 transformation matrix
    pose = torch.eye(4)
    pose[:3, :3] = rotation  # Rotation part
    pose[:3, 3] = position  # Camera position in world coordinates

    return pose


def train(checkpoint_path, n_iters=1000):
    model = NeRF().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.005, eps=1e-4, weight_decay=0.001
    )

    guide = StableGuide(
        t_range=(0.02, 0.98), guidance_scale=1, device=device, dtype=torch.float16
    )

    prompt = "a high quality very zoomed out photo of a red apple"
    negative_prompt = "cropped, out of frame, morbid, mutilated, bizarre, corrupted, malformed, low quality, artifacts, watermark, signature"

    neg_embeds = guide.encode_text(negative_prompt)

    pos_embeds_top = guide.encode_text("overhead view of " + prompt)
    pos_embeds_front = guide.encode_text("front view of " + prompt)
    pos_embeds_side = guide.encode_text("side view of " + prompt)
    pos_embeds_back = guide.encode_text("back view of " + prompt)

    plot_step = 250
    n_samples = 64

    for i in tqdm(range(n_iters)):
        theta = torch.rand(1) * 100 - 10
        phi = torch.rand(1) * 360
        pose = get_camera_pose(theta, phi, 4).to(device)

        rays_o, rays_d = model.define_rays(height, width, focal, pose)
        rgb = resize(
            (
                model.render(
                    model,
                    rays_o,
                    rays_d,
                    near=2,
                    far=4,
                    n_samples=n_samples,
                    rand=True,
                )[0]
                .permute(2, 0, 1)
                .view(1, 3, height, width)
            ),
            (512, 512),
        ).half()

        curr_embeds = None
        if phi > 60:
            curr_embeds = pos_embeds_top
        elif theta <= 45 and theta > 315:
            curr_embeds = pos_embeds_front
        elif theta > 135 and theta < 225:
            curr_embeds = pos_embeds_back
        else:
            curr_embeds = pos_embeds_side

        optimizer.zero_grad()
        loss = guide.calculate_sds_loss(
            guide.encode_images(rgb), curr_embeds, neg_embeds
        )
        loss.backward()
        optimizer.step()

        if i % plot_step == 0:
            torch.save(model.state_dict(), f"{checkpoint_path}/ckpt{i}.pth")

            with torch.no_grad():
                rays_o, rays_d = model.define_rays(
                    height,
                    width,
                    focal,
                    get_camera_pose(torch.tensor(0), torch.tensor(0), 4).to(device),
                )
                test_rgb, test_depth = model.render(
                    model,
                    rays_o,
                    rays_d,
                    near=2,
                    far=4,
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
                plt.show()

    torch.save(model.state_dict(), f"{checkpoint_path}/ckpt_final.pth")


checkpoint = "output"
train(checkpoint)
# visualize(checkpoint)
