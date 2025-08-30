import numpy as np
import torch
from scipy.spatial.transform import Rotation as ScipyRotation

from nvtorchcam.cameras import PinholeCamera
from torch.nn.functional import grid_sample


def project_to_nfov_nvtorchcam(e_img, fov_deg_x=80, out_hw=(512, 512)):
    """
    Projects ERP image into NFoV views using ray sampling and grid_sample.
    This avoids broken or missing nvtorchcam functions.
    """
    h_out, w_out = out_hw
    fov_deg_y = fov_deg_x * h_out / w_out
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalize image
    e_img_tensor = torch.from_numpy(e_img).permute(2, 0, 1).unsqueeze(0).float() / 255.
    e_img_tensor = e_img_tensor.to(device)

    H_erp, W_erp = e_img.shape[:2]

    pitch_angles = [-45, 0, 45]
    yaw_angles = [-180, -90, 0, 90]
    angles = []
    images = []

    for pitch in pitch_angles:
        for yaw in yaw_angles:
            # Create pinhole intrinsics
            fov_x_rad = np.radians(fov_deg_x)
            fx = (w_out / 2) / np.tan(fov_x_rad / 2)
            fy = fx
            cx = w_out / 2
            cy = h_out / 2
            intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)

            # Create PinholeCamera
            cam = PinholeCamera.make(intrinsics).to(device)
            R = ScipyRotation.from_euler('yx', [yaw, -pitch], degrees=True).as_matrix()
            cam.R = torch.tensor(R, device=device, dtype=torch.float32)
            cam.T = torch.zeros(3, device=device, dtype=torch.float32)

            # Get rays (o, d)
            rays_o, rays_d, valid = cam.get_camera_rays((h_out, w_out), unit_vec=True)  # shape (H, W, 3)

            # Convert ray directions to spherical angles (θ = pitch, φ = yaw)
            x, y, z = rays_d.unbind(-1)
            theta = torch.arccos(y)  # [0, π]
            phi = torch.atan2(x, z)  # [-π, π]

            # Normalize to ERP coords (0–1 range)
            grid_u = phi / np.pi / 2 + 0.5  # [0,1]
            grid_v = theta / np.pi          # [0,1]

            # Convert to [-1, 1] for grid_sample
            grid_x = grid_u * 2 - 1
            grid_y = grid_v * 2 - 1
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

            # Sample ERP image
            sampled = grid_sample(
                input=e_img_tensor,
                grid=grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True
            )  # (1, 3, H, W)

            out_img = (sampled[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            images.append(out_img)
            angles.append((pitch, yaw))

    return images, angles
