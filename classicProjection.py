# projection.py

import numpy as np
import py360convert

def project_to_nfov_grid(e_img, fov_deg=(80, 80), out_hw=(512, 512)):
    """
    Projects an equirectangular image to a grid of perspective views (NFoV).

    Args:
        e_img (np.ndarray): The equirectangular input image (H, W, 3).
        fov_deg (tuple): The vertical and horizontal field of view for each view.
        out_hw (tuple): The desired output height and width for each view.

    Returns:
        list[np.ndarray]: A list of the projected perspective images.
        list[tuple]: A list of the (pitch, yaw) angles for each projection.
    """
    h_out, w_out = out_hw
    fov_v, fov_h = fov_deg
    
    # Define the grid of camera angles (pitch, yaw) in degrees
    # This 3x4 grid covers the full 360 view with some overlap
    pitch_angles = [-45, 0, 45]
    yaw_angles = [-180, -90, 0, 90]

    # Uncomment the following two lines for more images.
    # pitch_angles = [-90, -45, 0, 45, 90]
    # yaw_angles = [-180, -135, -90, -45, 0, 45, 90, 135]
    
    nfov_images = []
    angles = []
    
    print(f"Generating {len(pitch_angles) * len(yaw_angles)} NFoV views...")
    for pitch in pitch_angles:
        for yaw in yaw_angles:
            # Project equirectangular to a perspective (NFoV) view
            p_img = py360convert.e2p(
                e_img,
                fov_deg=(fov_v, fov_h),
                u_deg=yaw,    # Yaw (left/right)
                v_deg=pitch,  # Pitch (up/down)
                out_hw=(h_out, w_out),
                in_rot_deg=0,
                mode='bilinear'
            )
            nfov_images.append(p_img)
            angles.append((pitch, yaw))
            
    return nfov_images, angles