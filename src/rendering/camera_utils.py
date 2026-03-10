import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras


def create_cameras(cfg, device):
    num_views = cfg["num_views"]
    dist = cfg["camera_distance"]

    elev = torch.linspace(
        cfg["elevation_range"][0],
        cfg["elevation_range"][1],
        num_views
    )

    azim = torch.linspace(
        cfg["azimuth_range"][0],
        cfg["azimuth_range"][1],
        num_views
    )

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    return cameras