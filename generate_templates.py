from camera_config import NEAR
from camera_config import FAR
from camera_config import FOV
import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import math
import matplotlib.pyplot as plt


def generate_templates(urdf_path, output_dir, num_views=8, image_size=(512, 384)):
    """
    Generate template images of the racecar using PyBullet.
    """
    # Connect to PyBullet in DIRECT mode (headless)
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load the racecar
    # Adjust base position if needed
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    racecarId = p.loadURDF(urdf_path, startPos, startOrientation)

    # Get AABB to center the camera
    aabb_min, aabb_max = p.getAABB(racecarId)
    center = [(aabb_max[i] + aabb_min[i]) / 2 for i in range(3)]
    extent = [aabb_max[i] - aabb_min[i] for i in range(3)]
    max_extent = max(extent)

    print(f"Object center: {center}")
    print(f"Object max extent: {max_extent}")

    os.makedirs(output_dir, exist_ok=True)

    # Camera settings

    # Distance from the target object
    # Adjust distance to fit object in view
    dist = max_extent * 2.0

    # Target position (center of the car)
    target_pos = center

    print(f"Generating {num_views} templates...")

    for i in range(num_views):
        azim = (i / num_views) * 360.0
        elev = 30.0  # Elevation angle

        # Convert spherical to cartesian coordinates for camera position
        rad_azim = math.radians(azim)
        rad_elev = math.radians(elev)

        cam_x = target_pos[0] + dist * math.cos(rad_elev) * math.cos(rad_azim)
        cam_y = target_pos[1] + dist * math.cos(rad_elev) * math.sin(rad_azim)
        cam_z = target_pos[2] + dist * math.sin(rad_elev)

        cam_pos = [cam_x, cam_y, cam_z]

        # Up vector (z-up world)
        up_vector = [0, 0, 1]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector,
        )

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=FOV, aspect=image_size[0] / image_size[1], nearVal=NEAR, farVal=FAR
        )

        # Render image
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=image_size[0],
            height=image_size[1],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        # rgbImg is [width, height, 4] (RGBA)
        rgb_array = np.array(rgbImg, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (height, width, 4))

        # depthImg is [width, height] floats
        depth_buffer = np.array(depthImg, dtype=np.float32)
        depth_buffer = np.reshape(depth_buffer, (height, width))

        # Convert depth buffer to actual depth values
        # PyBullet depth buffer is non-linear:
        # depth = far * near / (far - (far - near) * depth_buffer)
        real_depth = FAR * NEAR / (FAR - (FAR - NEAR) * depth_buffer)

        # Remove alpha channel for saving if desired, or keep it
        rgb_image = rgb_array[:, :, :3]

        filename = os.path.join(output_dir, f"template_{i:03d}_azim_{azim:.1f}.png")
        plt.imsave(filename, rgb_image)

        # Save depth image (normalized for visualization)
        depth_filename = os.path.join(output_dir, f"template_{i:03d}_depth.png")
        plt.imsave(depth_filename, real_depth, cmap="gray")

        # Save matrices and raw depth
        pose_filename = os.path.join(output_dir, f"template_{i:03d}_pose.npz")
        np.savez(
            pose_filename,
            view_matrix=view_matrix,
            projection_matrix=projection_matrix,
            azim=azim,
            elev=elev,
            dist=dist,
            depth=real_depth,
        )

        print(f"Saved {filename}")

    p.disconnect()


if __name__ == "__main__":
    urdf_path = "my_racecar/my_racecar.urdf"
    output_dir = "templates"

    generate_templates(urdf_path, output_dir)
