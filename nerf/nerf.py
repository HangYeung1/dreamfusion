import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class NeRF(nn.Module):
    def __init__(self, filter_size=128, num_encoding_functions=6):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size))
        self.layers.append(nn.Linear(filter_size, filter_size))
        self.output = nn.Linear(filter_size, 4)

    def init_weights(self):
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
        torch.nn.init.xavier_normal_(self.output.weight)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)

    def define_rays(self, height, width, focal, pose):
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
        )
        i = i.t()
        j = j.t()

        dirs = torch.stack(
            [
                (i - width * 0.5) / focal,
                -(j - height * 0.5) / focal,
                -torch.ones_like(i),
            ],
            -1,
        ).to(device)

        rays_d = torch.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)

        rays_o = pose[:3, -1].expand(rays_d.shape)

        return rays_o, rays_d

    def pos_encode(self, x, num_levels=6):
        ans = [x]
        for i in range(num_levels):
            ans.extend([a(i * x * 2 * np.pi) for a in [torch.sin, torch.cos]])
        return torch.cat(ans, -1)

    def calc_prodcum(self, tensor: torch.Tensor):
        cumprod = torch.roll(torch.cumprod(tensor, -1), 1, -1)
        cumprod[..., 0] = 1.0
        return cumprod

    def render(self, model, rays_o, rays_d, near, far, n_samples, rand=False):
        z = torch.linspace(near, far, n_samples).to(device)
        if rand:
            mids = 0.5 * (z[..., 1:] + z[..., :-1])
            upper = torch.cat([mids, z[..., -1:]], -1)
            lower = torch.cat([z[..., :1], mids], -1)
            t_rand = torch.rand(z.shape).to(device)
            z = lower + (upper - lower) * t_rand

        points = rays_o[..., None, :] + rays_d[..., None, :] * z[..., :, None]

        flat_points = torch.reshape(points, [-1, points.shape[-1]])
        flat_points = self.pos_encode(flat_points)

        chunk = 1024 * 32
        raw = torch.cat(
            [
                model(flat_points[i : i + chunk])
                for i in range(0, flat_points.shape[0], chunk)
            ],
            0,
        )
        raw = torch.reshape(raw, list(points.shape[:-1]) + [4])

        sigma = F.relu(raw[..., 3])
        rgb = torch.sigmoid(raw[..., :3])
        one_e_10 = torch.tensor([1e10], dtype=rays_o.dtype).to(device)
        dists = torch.cat(
            (z[..., 1:] - z[..., :-1], one_e_10.expand(z[..., :1].shape)), dim=-1
        )
        alpha = 1.0 - torch.exp(-sigma * dists)
        weights = alpha * self.calc_prodcum(1.0 - alpha + 1e-10)

        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)

        depth_map = torch.sum(weights * z, dim=-1)

        return rgb_map, depth_map
