import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.animation as animation

rawData = np.load("tiny_nerf_data.npz",allow_pickle=True)
images = rawData["images"]
poses = rawData["poses"]
focal = rawData["focal"]
H, W = images.shape[1:3]
H = int(H)
W = int(W)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testimg, testpose = images[99], poses[99]
plt.imshow(testimg)
plt.title('Dataset example')
plt.show()
images = torch.Tensor(images).to(device)
poses = torch.Tensor(poses).to(device)
testimg = torch.Tensor(testimg).to(device)
testpose = torch.Tensor(testpose).to(device)

def get_rays(H, W, focal, pose):
  """
  This function generates camera rays for each pixel in an image. It calculates the origin and direction of rays
  based on the camera's intrinsic parameters (focal length) and extrinsic parameters (pose).
  The rays are generated in world coordinates, which is crucial for the NeRF rendering process.

  Parameters:
  H (int): Height of the image in pixels.
  W (int): Width of the image in pixels.
  focal (float): Focal length of the camera.
  pose (torch.Tensor): Camera pose matrix of size 4x4.

  Returns:
  tuple: A tuple containing two elements:
      rays_o (torch.Tensor): Origins of the rays in world coordinates.
      rays_d (torch.Tensor): Directions of the rays in world coordinates.
  """
  # Create a meshgrid of image coordinates (i, j) for each pixel in the image.
  i, j = torch.meshgrid(
      torch.arange(W, dtype=torch.float32),
      torch.arange(H, dtype=torch.float32)
  )
  i = i.t()
  j = j.t()

  # Calculate the direction vectors for each ray originating from the camera center.
  # We assume the camera looks towards -z.
  # The coordinates are normalized with respect to the focal length.
  dirs = torch.stack(
      [(i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)], -1
      ).to(device)

  # Transform the direction vectors (dirs) from camera coordinates to world coordinates.
  # This is done using the rotation part (first 3 columns) of the pose matrix.
  rays_d = torch.sum(dirs[..., np.newaxis, :] * pose[:3, :3], -1)

  # The ray origins (rays_o) are set to the camera position, given by the translation part (last column) of the pose matrix.
  # The position is expanded to match the shape of rays_d for broadcasting.
  rays_o = pose[:3, -1].expand(rays_d.shape)

  # Return the origins and directions of the rays.
  return rays_o, rays_d


def positional_encoder(x, L_embed=6):
  """
  This function applies positional encoding to the input tensor. Positional encoding is used in NeRF
  to allow the model to learn high-frequency details more effectively. It applies sinusoidal functions
  at different frequencies to the input.

  Parameters:
  x (torch.Tensor): The input tensor to be positionally encoded.
  L_embed (int): The number of frequency levels to use in the encoding. Defaults to 6.

  Returns:
  torch.Tensor: The positionally encoded tensor.
  """

  # Initialize a list with the input tensor.
  rets = [x]

  # Loop over the number of frequency levels.
  for i in range(L_embed):
    #############################################################################
    #                                   TODO                                    #
    #############################################################################
    rets.extend([f(i*x*2*np.pi) for f in [torch.sin, torch.cos]])
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


  # Concatenate the original and encoded features along the last dimension.
  return torch.cat(rets, -1)

def cumprod_exclusive(tensor: torch.Tensor):
  """
  Compute the exclusive cumulative product of a tensor along its last dimension.
  'Exclusive' means that the cumulative product at each element does not include the element itself.
  This function is used in volume rendering to compute the product of probabilities
  along a ray, excluding the current sample point.

  Parameters:
  tensor (torch.Tensor): The input tensor for which to calculate the exclusive cumulative product.

  Returns:
  torch.Tensor: The tensor after applying the exclusive cumulative product.
  """

  # Compute the cumulative product along the last dimension of the tensor.
  cumprod = torch.cumprod(tensor, -1)

  # Roll the elements along the last dimension by one position.
  # This shifts the cumulative products to make them exclusive.
  cumprod = torch.roll(cumprod, 1, -1)

  # Set the first element of the last dimension to 1, as the exclusive product of the first element is always 1.
  cumprod[..., 0] = 1.

  return cumprod

class VeryTinyNerfModel(torch.nn.Module):
  """
  A very small implementation of a Neural Radiance Field (NeRF) model. This model is a simplified
  version of the standard NeRF architecture, it consists of a simple feedforward neural network with three linear layers.

  Parameters:
  filter_size (int): The number of neurons in the hidden layers. Default is 128.
  num_encoding_functions (int): The number of sinusoidal encoding functions. Default is 6.
  """

  def __init__(self, filter_size=128, num_encoding_functions=6):
    super(VeryTinyNerfModel, self).__init__()
    self.layer1 = torch.nn.Linear(3 + 3 * 2 * num_encoding_functions, filter_size)
    self.layer2 = torch.nn.Linear(filter_size, filter_size)
    self.layer3 = torch.nn.Linear(filter_size, 4)
    self.relu = torch.nn.functional.relu

  def forward(self, x):
    x = self.relu(self.layer1(x))
    x = self.relu(self.layer2(x))
    x = self.layer3(x)
    return x

def render(model, rays_o, rays_d, near, far, n_samples, rand=False):
  """
  Render a scene using a Neural Radiance Field (NeRF) model. This function samples points along rays,
  evaluates the NeRF model at these points, and applies volume rendering techniques to produce an image.

  Parameters:
  model (torch.nn.Module): The NeRF model to be used for rendering.
  rays_o (torch.Tensor): Origins of the rays.
  rays_d (torch.Tensor): Directions of the rays.
  near (float): Near bound for depth sampling along the rays.
  far (float): Far bound for depth sampling along the rays.
  n_samples (int): Number of samples to take along each ray.
  rand (bool): If True, randomize sample depths. Default is False.

  Returns:
  tuple: A tuple containing the RGB map and depth map of the rendered scene.
  """

  # Sample points along each ray, from 'near' to 'far'.
  z = torch.linspace(near, far, n_samples).to(device)
  if rand:
    mids = 0.5 * (z[..., 1:] + z[..., :-1])
    upper = torch.cat([mids, z[..., -1:]], -1)
    lower = torch.cat([z[..., :1], mids], -1)
    t_rand = torch.rand(z.shape).to(device)
    z = lower + (upper - lower) * t_rand

  #############################################################################
  #                                   TODO                                    #
  #############################################################################
  # Compute 3D coordinates of the sampled points along the rays.
  points = rays_o[..., None, :] + rays_d[..., None, :] * z[..., :, None]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  # Flatten the points and apply positional encoding.
  flat_points = torch.reshape(points, [-1, points.shape[-1]])
  flat_points = positional_encoder(flat_points)

  # Evaluate the model on the encoded points in chunks to manage memory usage.
  chunk = 1024 * 32
  raw = torch.cat([model(flat_points[i:i + chunk]) for i in range(0, flat_points.shape[0], chunk)], 0)
  raw = torch.reshape(raw, list(points.shape[:-1]) + [4])

  # Compute densities (sigmas) and RGB values from the model's output.
  sigma = F.relu(raw[..., 3])
  rgb = torch.sigmoid(raw[..., :3])

  # Perform volume rendering to obtain the weights of each point.
  one_e_10 = torch.tensor([1e10], dtype=rays_o.dtype).to(device)
  dists = torch.cat((z[..., 1:] - z[..., :-1], one_e_10.expand(z[..., :1].shape)), dim=-1)
  alpha = 1. - torch.exp(-sigma * dists)
  weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

  #############################################################################
  #                                   TODO                                    #
  #############################################################################
  # Compute the weighted sum of RGB values along each ray to get the final pixel color.
  rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)
  
  # Compute the depth map as the weighted sum of sampled depths.
  depth_map = torch.sum(weights*z, dim=-1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return rgb_map, depth_map

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.])).to(device)

def train(model, optimizer, n_iters=3000):
  """
  Train the Neural Radiance Field (NeRF) model. This function performs training over a specified number of iterations,
  updating the model parameters to minimize the difference between rendered and actual images.

  Parameters:
  model (torch.nn.Module): The NeRF model to be trained.
  optimizer (torch.optim.Optimizer): The optimizer used for training the model.
  n_iters (int): The number of iterations to train the model. Default is 3000.
  """

  psnrs = []
  iternums = []

  plot_step = 500
  n_samples = 64   # Number of samples along each ray.

  for i in tqdm(range(n_iters + 1)):
    # Randomly select an image from the dataset and use it as the target for training.
    images_idx = np.random.randint(images.shape[0])
    target = images[images_idx]
    pose = poses[images_idx]

    # Compute the rays for the selected image.
    rays_o, rays_d = get_rays(H, W, focal, pose)
    # Render the scene using the current model state.
    rgb, depth = render(model, rays_o, rays_d, near=2., far=6., n_samples=n_samples, rand=True)

    optimizer.zero_grad()
    image_loss = torch.nn.functional.mse_loss(rgb, target)
    image_loss.backward()
    optimizer.step()

    if i % plot_step == 0:
      torch.save(model.state_dict(), 'ckpt.pth')
      # Render a test image to evaluate the current model performance.
      with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth = render(model, rays_o, rays_d, near=2., far=6., n_samples=n_samples)
        loss = torch.nn.functional.mse_loss(rgb, testimg)
        # Calculate PSNR for the rendered image.
        psnr = mse2psnr(loss)

        psnrs.append(psnr.detach().cpu().numpy())
        iternums.append(i)

        # Plotting the rendered image and PSNR over iterations.
        plt.figure(figsize=(9, 3))

        plt.subplot(131)
        picture = rgb.cpu()  # Copy the rendered image from GPU to CPU.
        plt.imshow(picture)
        plt.title(f'RGB Iter {i}')

        plt.subplot(132)
        picture = depth.cpu() * (rgb.cpu().mean(-1)>1e-2)
        plt.imshow(picture, cmap='gray')
        plt.title(f'Depth Iter {i}')

        plt.subplot(133)
        plt.plot(iternums, psnrs)
        plt.title('PSNR')
        plt.show()

def render_rotating_nerf(model, H, W, focal, poses, n_samples, near=2., far=6., n_frames=120):
    """
    Render a rotating NeRF scene by generating images from different viewpoints.

    Parameters:
    model (torch.nn.Module): The NeRF model to be used for rendering.
    H (int): Height of the image in pixels.
    W (int): Width of the image in pixels.
    focal (float): Focal length of the camera.
    poses (torch.Tensor): Camera poses.
    n_samples (int): Number of samples to take along each ray.
    near (float): Near bound for depth sampling along the rays.
    far (float): Far bound for depth sampling along the rays.
    n_frames (int): Number of frames to render for the rotation.

    Returns:
    list: A list of rendered images.
    """
    rendered_images = []
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        rotation_matrix = torch.tensor([
            [np.cos(angle), -np.sin(angle), 0, 0],
            [np.sin(angle), np.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32).to(device)
        pose = torch.matmul(rotation_matrix, poses[0])
        rays_o, rays_d = get_rays(H, W, focal, pose)
        rgb, _ = render(model, rays_o, rays_d, near, far, n_samples)
        rendered_images.append(rgb.cpu().numpy())
    return rendered_images

def display_renderings():
    nerf = VeryTinyNerfModel()
    nerf = nn.DataParallel(nerf).to(device)
    ckpt = torch.load('ckpt.pth')
    nerf.load_state_dict(ckpt)
    test_img_idx_list = [0, 40, 80]
    H, W = 100, 100
    with torch.no_grad():
        for test_img_idx in test_img_idx_list:
            rays_o, rays_d = get_rays(H, W, focal, poses[test_img_idx])
            rgb, depth = render(nerf, rays_o, rays_d, near=2., far=6., n_samples=64)
            plt.figure(figsize=(9,3))

            plt.subplot(131)
            picture = rgb.cpu()
            plt.title("RGB Prediction #{}".format(test_img_idx))
            plt.imshow(picture)

            plt.subplot(132)
            picture = depth.cpu() * (rgb.cpu().mean(-1)>1e-2)
            plt.imshow(picture, cmap='gray')
            plt.title("Depth Prediction #{}".format(test_img_idx))

            plt.subplot(133)
            plt.title("Ground Truth #{}".format(test_img_idx))
            plt.imshow(rawData["images"][test_img_idx])
            plt.show()

        # Render rotating NeRF
        rotating_images = render_rotating_nerf(nerf, H, W, focal, poses, n_samples=64)

        fig = plt.figure()
        im = plt.imshow(rotating_images[0])

        def update(frame):
            im.set_array(rotating_images[frame])
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=len(rotating_images), interval=1000/30)
        ani.save('rotating_video.mp4', writer='ffmpeg', fps=30)
        plt.show()

# nerf = VeryTinyNerfModel()
# nerf = nn.DataParallel(nerf).to(device)
# optimizer = torch.optim.Adam(nerf.parameters(), lr=5e-3, eps = 1e-7)
# train(nerf, optimizer)
display_renderings()




