## feature_matcher_tools.py

# Stdlib
import os
import sys
import math
import shutil
from pathlib import Path
from typing import Optional, Tuple, Literal, Dict, Any

# Third-party
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import imageio
import requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm.notebook import tqdm
from skimage import img_as_ubyte

## Imports (from inside the mast3r directory)

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images


# visualize a few matches
import numpy as np
import torch
import torchvision.transforms.functional
from matplotlib import pyplot as pl

import cv2


# import unproject_3d_from_depth_tools as unprojtools


class FeatureMatcher:
    @staticmethod
    def filter_matches_by_mask(matches_im0, matches_im1, mask0, mask1):
        """
        Filters out matches where either point lies in the background of its respective mask.

        Args:
            matches_im0 (np.ndarray): Nx2 array of (x, y) points in image 0.
            matches_im1 (np.ndarray): Nx2 array of (x, y) points in image 1.
            mask0 (np.ndarray): Binary mask for image 0 (H, W), non-zero = foreground.
            mask1 (np.ndarray): Binary mask for image 1 (H, W), non-zero = foreground.

        Returns:
            filtered_im0 (np.ndarray): Filtered matches from image 0.
            filtered_im1 (np.ndarray): Corresponding filtered matches from image 1.
        """
        # Bounds check
        H0, W0 = mask0.shape
        H1, W1 = mask1.shape

        x0, y0 = matches_im0[:, 0], matches_im0[:, 1]
        x1, y1 = matches_im1[:, 0], matches_im1[:, 1]

        valid0 = (x0 >= 0) & (x0 < W0) & (y0 >= 0) & (y0 < H0)
        valid1 = (x1 >= 0) & (x1 < W1) & (y1 >= 0) & (y1 < H1)
        safe = valid0 & valid1

        matches_im0 = matches_im0[safe]
        matches_im1 = matches_im1[safe]
        x0, y0 = matches_im0[:, 0], matches_im0[:, 1]
        x1, y1 = matches_im1[:, 0], matches_im1[:, 1]

        # Foreground mask check
        in_fg0 = mask0[y0, x0] > 0
        in_fg1 = mask1[y1, x1] > 0
        valid = in_fg0 & in_fg1

        return matches_im0[valid], matches_im1[valid]

    @staticmethod
    def mast3r_inference(images, model, device):
        """
        Return values:
            matches_im0, matches_im1, view1, pred1, view2, pred2


        """

        # Inference
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1, desc2 = (
            pred1["desc"].squeeze(0).detach(),
            pred2["desc"].squeeze(0).detach(),
        )

        # find 2D-2D matches between the two images
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=8,
            device=device,
            dist="dot",
            block_size=2**13,
        )

        # ignore small border around the edge
        H0, W0 = view1["true_shape"][0]
        valid_matches_im0 = (
            (matches_im0[:, 0] >= 3)
            & (matches_im0[:, 0] < int(W0) - 3)
            & (matches_im0[:, 1] >= 3)
            & (matches_im0[:, 1] < int(H0) - 3)
        )

        H1, W1 = view2["true_shape"][0]
        valid_matches_im1 = (
            (matches_im1[:, 0] >= 3)
            & (matches_im1[:, 0] < int(W1) - 3)
            & (matches_im1[:, 1] >= 3)
            & (matches_im1[:, 1] < int(H1) - 3)
        )

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = (
            matches_im0[valid_matches],
            matches_im1[valid_matches],
        )

        return matches_im0, matches_im1, view1, pred1, view2, pred2

    @staticmethod
    def mast3r_view2rgbimage(view):
        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device="cpu").reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device="cpu").reshape(1, 3, 1, 1)

        rgb_tensor = view["img"] * image_std + image_mean
        rgb_image = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        return rgb_image

    @staticmethod
    def visualize_mast3r_matches(matches_im0, matches_im1, view1, view2, n_viz_lines):
        n_viz = n_viz_lines
        num_matches = matches_im0.shape[0]
        match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = (
            matches_im0[match_idx_to_viz],
            matches_im1[match_idx_to_viz],
        )

        # Make a list of rgb images from view1 and view2
        viz_imgs = []
        for i, view in enumerate([view1, view2]):
            # convert the view output from mast3r to standard rgb image
            rgb_image = FeatureMatcher.mast3r_view2rgbimage(view)
            # Append rgb image to list
            viz_imgs.append(rgb_image)

        # Compose a two-image view with views arranged side by side.
        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        img0 = np.pad(
            viz_imgs[0],
            ((0, max(H1 - H0, 0)), (0, 0), (0, 0)),
            "constant",
            constant_values=0,
        )
        img1 = np.pad(
            viz_imgs[1],
            ((0, max(H0 - H1, 0)), (0, 0), (0, 0)),
            "constant",
            constant_values=0,
        )
        img = np.concatenate((img0, img1), axis=1)

        # Show side-by-side images first
        pl.figure()
        pl.axis("off")
        pl.imshow(img)

        # Draw match lines in color
        cmap = pl.get_cmap("jet")
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot(
                [x0, x1 + W0],
                [y0, y1],
                "-+",
                color=cmap(i / (n_viz - 1)),
                scalex=False,
                scaley=False,
            )

        # pl.savefig("output_mast3r_christ.jpg", dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        pl.show(block=True)

    @staticmethod
    def smooth_and_fill_mask(mask):
        """
        Smooth and fill holes in a binary mask (NumPy array).

        Args:
            mask (np.ndarray): Binary mask (H, W), values 0 or 1 (or 0/255).

        Returns:
            np.ndarray: Smoothed and hole-filled binary mask (H, W), dtype=uint8
        """
        # Ensure mask is in 0/255 format
        mask = (mask > 0).astype(np.uint8) * 255

        # Smoothing: morphological closing (dilate then erode)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Hole filling via flood fill
        h, w = smoothed.shape
        flood_fill = smoothed.copy()
        mask_flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(flood_fill, mask_flood, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood_fill)
        filled = cv2.bitwise_or(smoothed, flood_inv)

        # Apply Gaussian blur to smooth edges
        filled = cv2.GaussianBlur(filled, (3, 3), 0)

        # Return binary mask (0 or 1)
        return (filled > 0).astype(np.uint8)

    @staticmethod
    def create_masks(input_dir, output_dir):
        """
        Converts a sequence of PNG masks into binary masks (foreground = 1, background = 0).
        Saves them into a new directory.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # List PNG files
        mask_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
        )

        for fname in mask_files:
            path = os.path.join(input_dir, fname)
            mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            if mask is None:
                print(f"Warning: Failed to read {path}")
                continue

            # Convert to grayscale if needed
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Consolidate to binary: foreground = 1, background = 0
            binary_mask = (mask > 0).astype(np.uint8) * 255

            # Save result
            save_path = os.path.join(output_dir, fname)
            cv2.imwrite(save_path, binary_mask)

        print(f"[INFO] Saved binary masks to: {output_dir}")

    @staticmethod
    def filter_nan_points(
        X_world: np.ndarray, points_uv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter out rows where either X_world (3D) or points_uv (2D)
        contain NaN/Inf, and return the cleaned arrays + index tracking.

        Args:
            X_world: (N,3) NumPy array of 3D coordinates.
            points_uv: (N,2) NumPy array of 2D coordinates (aligned with X_world).

        Returns:
            X_world_clean: (M,3) array with only valid rows.
            points_uv_clean: (M,2) array with only valid rows.
            kept_indices: (M,) indices of rows kept.
            removed_indices: (K,) indices of rows removed.
        """
        if X_world.shape[0] != points_uv.shape[0]:
            raise ValueError("X_world and points_uv must have the same number of rows")

        # Mask: valid only if both 3D and 2D rows are finite
        mask_3d = np.all(np.isfinite(X_world), axis=1)
        # mask_2d = np.all(np.isfinite(points_uv), axis=1)
        # mask = mask_3d & mask_2d
        mask = mask_3d

        kept_indices = np.where(mask)[0]
        removed_indices = np.where(~mask)[0]

        X_world_clean = X_world[mask]
        points_uv_clean = points_uv[mask]

        return X_world_clean, points_uv_clean, kept_indices, removed_indices


# import numpy as np
# import torch
# import cv2
# import matplotlib.pyplot as plt

# try:
#     from pytorch3d.renderer import (
#         PerspectiveCameras,
#         MeshRenderer,
#         MeshRasterizer,
#         SoftPhongShader,
#         RasterizationSettings,
#         PointLights,
#     )
#     from pytorch3d.utils import cameras_from_opencv_projection
# except ImportError:
#     print("Warning: PyTorch3D not found. Some functions may not work.")
#     PerspectiveCameras = None
#     MeshRenderer = None
#     MeshRasterizer = None
#     SoftPhongShader = None
#     RasterizationSettings = None
#     PointLights = None
#     cameras_from_opencv_projection = None

# import unproject_3d_from_depth_tools as unprojtools


# def estimate_pose_pnp(
#     mesh,
#     obj_pts3d,  # (M,3) torch or np (object/world coords)
#     img_pts2d,  # (M,2) torch or np (pixel coords; origin top-left)
#     fx,
#     fy,
#     cx,
#     cy,  # intrinsics (pixels)
#     W,
#     H,  # image size (pixels)
#     # base_rgb=None,              # np image to overlay wireframe on (optional)
#     # wireframe_pts3d=None,       # (N,3) torch or np for wireframe (e.g., all cube corners)
#     # wireframe_edges=None,       # list of (i,j) edges over wireframe_pts3d indices
#     ransac=True,  # use solvePnPRansac for robustness
#     refine=True,  # refine with LM on inliers
#     reproj_err=2.0,  # RANSAC reprojection error (px)
#     iters=1000,  # RANSAC iterations
#     pnp_flag=None,  # override OpenCV flag; default chooses AP3P/EPNP
# ):
#     """
#     Returns:
#         result: dict with keys:
#           R_p3d (1,3,3) torch, T_p3d (1,3) torch, inliers (np Nx1) or None,
#           rms_px (float), cam_rec (PerspectiveCameras),
#           rgb_rec (HxWx3 np float32 in [0,1]),ß
#           proj_wireframe (dict with 'uv' Nx2 and 'z' N arrays) if wireframe provided.
#     """
#     # ---------- Prepare data ----------
#     # Choose a single device; use the mesh's device as authority
#     device = mesh.verts_packed().device

#     # to numpy float64 for OpenCV
#     obj_np = (
#         obj_pts3d.detach().cpu().numpy()
#         if isinstance(obj_pts3d, torch.Tensor)
#         else np.asarray(obj_pts3d)
#     )
#     img_np = (
#         img_pts2d.detach().cpu().numpy()
#         if isinstance(img_pts2d, torch.Tensor)
#         else np.asarray(img_pts2d)
#     )
#     obj_np = obj_np.astype(np.float64)
#     img_np = img_np.astype(np.float64)

#     assert obj_np.shape[0] >= 4 and obj_np.shape[0] == img_np.shape[0], (
#         "Need >=4 matching points and same count."
#     )

#     # Intrinsics for OpenCV
#     K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
#     dist = np.zeros(5, dtype=np.float64)  # no distortion

#     # Choose a PnP solver flag
#     if pnp_flag is None:
#         pnp_flag = cv2.SOLVEPNP_AP3P if obj_np.shape[0] == 4 else cv2.SOLVEPNP_EPNP

#     # ---------- PnP (RANSAC + refinement) ----------
#     if ransac:
#         ok, rvec, tvec, inliers = cv2.solvePnPRansac(
#             obj_np,
#             img_np,
#             K,
#             dist,
#             flags=pnp_flag,
#             reprojectionError=float(reproj_err),
#             iterationsCount=int(iters),
#         )
#         if not ok:
#             raise RuntimeError("solvePnPRansac failed to find a pose")
#         inl = inliers[:, 0] if inliers is not None else np.arange(len(obj_np))
#     else:
#         ok, rvec, tvec = cv2.solvePnP(obj_np, img_np, K, dist, flags=pnp_flag)
#         if not ok:
#             raise RuntimeError("solvePnP failed to find a pose")
#         inl = np.arange(len(obj_np))
#         inliers = None

#     if refine and inl.size >= 4:
#         rvec, tvec = cv2.solvePnPRefineLM(obj_np[inl], img_np[inl], K, dist, rvec, tvec)

#     # OpenCV pose from PnP

#     # Convert to rotation matrix
#     R_cv, _ = cv2.Rodrigues(rvec)  # (3,3)
#     t_cv = tvec.reshape(3)  # (3,)

#     # PyTorch3D row-vector convention matches OpenCV here:
#     # X_cam = X_world @ R_p3d^T + T_p3d  with  R_p3d = R_cv, T_p3d = t_cv^T
#     R_p3d = torch.tensor(R_cv, dtype=torch.float32, device=device).unsqueeze(
#         0
#     )  # (1,3,3)
#     T_p3d = torch.tensor(t_cv, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3)

#     # ---------- Reprojection error on inliers ----------
#     proj_inl, _ = cv2.projectPoints(obj_np[inl], rvec, tvec, K, dist)
#     proj_inl = proj_inl.squeeze(1)
#     rms_px = float(np.sqrt(np.mean(np.sum((proj_inl - img_np[inl]) ** 2, axis=1))))

#     result = {
#         "R_p3d": R_p3d,
#         "T_p3d": T_p3d,
#         "inliers": inliers,
#         "rms_px": rms_px,
#         # "cam_rec": cam_rec,
#         # "rgb_rec": rgb_rec,
#     }

#     return result


# ### Function to display the image and depth maps given the R T and camera (maybe we have that already somewhere)
# def show_pose_estimated_by_pnp(mesh, R_p3d, T_p3d, fx, fy, cx, cy, W, H):
#     # These is the pose estimated by PnP
#     R_pnp = R_p3d
#     T_pnp = T_p3d

#     # Create rgb and depth images
#     rgb_test_from_pnp, depth_test_from_pnp, cams_test_from_pnp = (
#         unprojtools.RenderWithPytorch3D.render_rgb_depth_from_view_from_RT(
#             mesh,
#             fx=fx,
#             fy=fy,
#             cx=cx,
#             cy=cy,
#             width=W,
#             height=H,
#             R=R_pnp,
#             T=T_pnp,
#         )
#     )

#     # Show
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(np.clip(rgb_test_from_pnp, 0, 1))
#     plt.axis("off")
#     plt.title("RGB")

#     # Depth visualization (treat -1 as invalid)
#     vis = unprojtools.ImageProcessor.depth_to_rgb(
#         depth_test_from_pnp, cmap="plasma", bg_mode="white"
#     )

#     plt.subplot(1, 2, 2)
#     plt.imshow(vis)
#     plt.axis("off")
#     plt.title("Depth")
#     plt.show()


# def show_rgb_depth_side_by_side(rgb, depth, cmap="plasma", bg_mode="white"):
#     # Show
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.imshow(np.clip(rgb, 0, 1))
#     plt.axis("off")
#     plt.title("RGB")

#     # Depth visualization (treat -1 as invalid)
#     vis = unprojtools.ImageProcessor.depth_to_rgb(depth, cmap="plasma", bg_mode="white")
#     plt.subplot(1, 2, 2)
#     plt.imshow(vis)
#     plt.axis("off")
#     plt.title("Depth")
#     plt.show()


# def set_initial_pose(mesh, distance, elev, azim, roll, fx, fy, cx, cy, W, H):
#     # # Camera extrinsics (look at origin)
#     # distance, elev, azim, roll  = 4, 20.0, -45.0, 0

#     # Create rgb and depth images
#     rgb, depth, cams = unprojtools.RenderWithPytorch3D.render_rgb_depth_from_view(
#         mesh,
#         fx=fx,
#         fy=fy,
#         cx=cx,
#         cy=cy,
#         width=W,
#         height=H,
#         distance=distance,
#         elev=elev,
#         azim=azim,
#         roll_deg=roll,
#         roll_mode="camera",  # try "camera" if you prefer
#     )

#     return rgb, depth, cams
