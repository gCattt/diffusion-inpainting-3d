import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras


def turntable_multi_ring(cfg, device):
    n_rings = cfg["turntable_rings"]
    n_views = cfg["views_per_ring"]

    elev_levels = [
        cfg["elev_mid"],
        cfg["elev_high"],
        cfg["elev_low"],
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
    dist = cfg["camera_distance"]
    fov = cfg["fov"]

    elev, azim = turntable_multi_ring(cfg, device)

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

    return cameras