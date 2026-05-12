import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras

# def fibonacci_sphere(n):
#     i = torch.arange(n)
#     phi = (1 + 5 ** 0.5) / 2

#     theta = 2 * torch.pi * i / phi
#     z = 1 - (2 * i + 1) / n
#     radius = torch.sqrt(1 - z * z)

#     x = radius * torch.cos(theta)
#     y = radius * torch.sin(theta)

#     elev = torch.rad2deg(torch.asin(z))
#     azim = torch.rad2deg(torch.atan2(y, x))

#     return elev, azim

def turntable_multi_ring(cfg, device):
    n_rings = cfg.get("turntable_rings", 3)
    n_views = cfg.get("views_per_ring", 8)

    elev_levels = [
        cfg.get("elev_mid", 0.0),
        cfg.get("elev_high", 30.0),
        cfg.get("elev_low", -30.0),
    ][:n_rings]

    azim = torch.linspace(0, 360, n_views + 1)[:-1]

    elev_all, azim_all = [], []

    for elev in elev_levels:
        elev_all.append(torch.full((n_views,), float(elev)))
        azim_all.append(azim)

    elev = torch.cat(elev_all)
    azim = torch.cat(azim_all)

    if cfg.get("add_top_bottom", True):
        elev = torch.cat([
            elev,
            torch.tensor([float(cfg.get("elev_top", 85.0))]),
            torch.tensor([float(cfg.get("elev_bottom", -85.0))]),
        ])
        azim = torch.cat([
            azim,
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        ])

    return elev.to(device), azim.to(device)

def create_cameras(cfg, device):
    # num_views = cfg["num_views"]
    dist = cfg["camera_distance"]
    fov = cfg.get("fov", 60.0)

    # elev, azim = fibonacci_sphere(num_views)

    elev, azim = turntable_multi_ring(cfg, device)

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    return cameras