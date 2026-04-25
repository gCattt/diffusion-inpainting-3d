import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras


def fibonacci_sphere(n):
    i = torch.arange(n)
    phi = (1 + 5 ** 0.5) / 2

    theta = 2 * torch.pi * i / phi
    z = 1 - (2 * i + 1) / n
    radius = torch.sqrt(1 - z * z)

    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    elev = torch.rad2deg(torch.asin(z))
    azim = torch.rad2deg(torch.atan2(y, x))

    return elev, azim

def create_cameras(cfg, device):
    num_views = cfg["num_views"]
    dist = cfg["camera_distance"]
    fov = cfg.get("fov", 60.0)

    elev, azim = fibonacci_sphere(num_views)

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    return cameras