import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pybullet as p
import pybullet_data
import math
import open3d as o3d
from scipy.spatial.transform import Rotation

# Add mast3r repo root to path so we can import mast3r and dust3r
sys.path.append(os.path.join(os.getcwd(), "mast3r"))

# Try to import mast3r
try:
    from feature_matcher_tools import FeatureMatcher
    from mast3r.model import AsymmetricMASt3R
    from dust3r.utils.image import load_images
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def load_model(device, checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    model = AsymmetricMASt3R.from_pretrained(checkpoint_path).to(device)
    return model


def get_intrinsics(fov, width, height):
    aspect = width / height
    # fov is vertical fov in degrees? PyBullet computeProjectionMatrixFOV takes fov in degrees.
    # But is it vertical or horizontal? Documentation says "fov: field of view angle in degrees". Usually vertical.
    # f_y = H / (2 * tan(fov/2))
    f_y = height / (2 * math.tan(math.radians(fov) / 2))
    f_x = f_y  # Assuming square pixels? Or aspect ratio?
    # If aspect != 1, then f_x = f_y * aspect? No, aspect = w/h.
    # If we want same FOV, then tan(fov_x/2) = aspect * tan(fov_y/2).
    # f_x = W / (2 * tan(fov_x/2)) = W / (2 * aspect * tan(fov_y/2)) = W / (2 * (W/H) * tan(fov_y/2)) = H / (2 * tan(fov_y/2)) = f_y.
    # So f_x = f_y if pixels are square.

    cx = width / 2
    cy = height / 2

    K = np.array([[f_x, 0, cx], [0, f_y, cy], [0, 0, 1]])
    return K


def unproject_points(uv, depth_map, K, view_matrix):
    """
    Unproject 2D points to 3D in World frame.
    uv: (N, 2) array of pixel coordinates
    depth_map: (H, W) array of depth values
    K: (3, 3) camera matrix
    view_matrix: (4, 4) view matrix (World -> Camera)
    """
    # Inverse camera matrix
    K_inv = np.linalg.inv(K)

    # Inverse view matrix (Camera -> World)
    view_inv = np.linalg.inv(view_matrix)

    points_3d = []
    valid_indices = []

    h, w = depth_map.shape

    for i, (u, v) in enumerate(uv):
        u_int, v_int = int(round(u)), int(round(v))

        if 0 <= u_int < w and 0 <= v_int < h:
            d = depth_map[v_int, u_int]

            # If depth is valid (PyBullet depth is far*near / ...) - we already converted it in generate_templates
            # Background is at far=5.0. We should filter it out.
            if d > 0 and d < 4.5:  # Filter background (far=5.0)
                # Unproject to Camera frame (OpenCV convention: +Z forward, +Y down)
                # P_cam = d * K_inv * [u, v, 1]
                p_img = np.array([u, v, 1.0])
                p_cam_cv = d * (K_inv @ p_img)

                # Convert to OpenGL Camera frame (+Z backward, +Y up)
                # x_gl = x_cv, y_gl = -y_cv, z_gl = -z_cv
                p_cam_gl = p_cam_cv * np.array([1, -1, -1])

                # Transform to World frame
                # P_world = view_inv * P_cam_gl
                p_cam_hom = np.append(p_cam_gl, 1.0)
                p_world_hom = view_inv @ p_cam_hom
                p_world = p_world_hom[:3] / p_world_hom[3]

                points_3d.append(p_world)
                valid_indices.append(i)

    return np.array(points_3d), valid_indices


def align_ground_plane(points_3d, target_up=np.array([0, 1, 0])):
    # Convert numpy → Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Fit plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.02, ransac_n=3, num_iterations=2000
    )

    normal = np.array(plane_model[:3], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # Compute rotation bringing normal → target_up
    v = np.cross(normal, target_up)
    c = np.dot(normal, target_up)
    s = np.linalg.norm(v)

    if s < 1e-6:
        # Already aligned
        return np.eye(3), points_3d

    # Rodrigues rotation
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

    # Apply rotation to point cloud
    aligned_points = (R @ points_3d.T).T

    return R, aligned_points


def visualize_pose(target_image_path, urdf_path, rvec, tvec, K, width, height):
    """
    Visualize the estimated pose by rendering the object in PyBullet.
    """
    # Connect to PyBullet
    # Use GUI to see it, or DIRECT to save image
    # Let's use DIRECT and save an overlay
    if p.isConnected():
        p.disconnect()
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Load object
    # We need to place the object at the estimated pose.
    # rvec, tvec are Object -> Camera transform (from solvePnP).
    # i.e. P_cam = R * P_obj + T

    # PyBullet needs World -> Object transform to place the object.
    # We can assume Camera is at World Origin (Identity).
    # Then Object pose in World is same as Object pose in Camera?
    # No, if Camera is at Origin, then P_cam = P_world.
    # So P_world = R * P_obj + T.
    # This means the object's position is T, and orientation is R.
    # Wait, solvePnP gives transform from Object frame to Camera frame.
    # If we place Camera at (0,0,0) looking down +Z (or -Z?), then we can place Object at (T, R).

    # PyBullet Camera:
    # We can set the camera view matrix to be Identity (or whatever matches our K convention).
    # PyBullet camera looks down -Z?
    # Let's check K convention. OpenCV uses +Z forward, +Y down, +X right.
    # PyBullet usually uses OpenGL convention: -Z forward, +Y up.

    # We need to be careful with coordinate systems.
    # solvePnP assumes OpenCV camera frame.

    # Let's place the object at (0,0,0) and move the camera?
    # Or place the camera at (0,0,0) and move the object?
    # Moving object is easier in PyBullet (resetBasePositionAndOrientation).

    # If P_cam = R_cv * P_obj + T_cv
    # And we want to render this in PyBullet.
    # We need to set PyBullet camera such that it matches OpenCV camera frame.
    # And set Object pose such that it matches R_cv, T_cv.

    # Convert OpenCV pose to PyBullet pose.
    # OpenCV: +Z forward, +Y down.
    # PyBullet (OpenGL): +X forward, +Z up.
    # Transform from OpenCV to OpenGL: Rotate 180 deg around X.
    # T_gl_cv = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    # P_gl = T_gl_cv * P_cv
    # P_gl = T_gl_cv * (R_cv * P_obj + T_cv)
    #      = (T_gl_cv * R_cv) * P_obj + (T_gl_cv * T_cv)

    # So R_gl = T_gl_cv * R_cv
    #    T_gl = T_gl_cv * T_cv

    R_cv, _ = cv2.Rodrigues(rvec)
    T_cv = tvec.flatten()

    # T_gl_cv matrix
    M_gl_cv = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    R_gl = M_gl_cv @ R_cv
    T_gl = M_gl_cv @ T_cv

    # Now we have Object pose in OpenGL camera frame.
    # If we place the Camera at Origin looking down -Z, then Object pose is (T_gl, R_gl).

    # Load object
    racecarId = p.loadURDF(urdf_path, [0, 0, 0])

    # Set object pose
    # PyBullet expects Quaternion for orientation
    # R_gl is rotation matrix. Convert to Quaternion.
    # We need a helper or use scipy.spatial.transform.Rotation
    # Or p.getQuaternionFromEuler(p.getEulerFromMatrix(...)) - PyBullet doesn't have getEulerFromMatrix exposed directly usually?
    # Actually, we can just use the matrix if we construct the transform.
    # But resetBasePositionAndOrientation takes quaternion.

    # Let's use scipy if available, or a simple conversion.

    quat = Rotation.from_matrix(R_gl).as_quat()  # (x, y, z, w)

    print(T_gl, quat)
    p.resetBasePositionAndOrientation(racecarId, T_gl, quat)

    # Set Camera
    # Camera at origin, looking down +X, up is +Z.
    cam_pos = [0, 0, 0]
    cam_target = [1, 0, 0]
    cam_up = [0, 0, 1]

    view_matrix = p.computeViewMatrix(cam_pos, cam_target, cam_up)

    # Projection matrix
    # We need to match K.
    # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    # OpenGL projection matrix:
    # [[2*fx/W, 0, 1 - 2*cx/W, 0],
    #  [0, 2*fy/H, 2*cy/H - 1, 0],
    #  [0, 0, -(f+n)/(f-n), -2*f*n/(f-n)],
    #  [0, 0, -1, 0]]
    # (Assuming standard glFrustum)

    near = 0.02
    far = 10.0

    P = np.zeros(16)
    P[0] = 2 * K[0, 0] / width
    P[5] = 2 * K[1, 1] / height
    P[8] = 1 - 2 * K[0, 2] / width
    P[9] = (
        2 * K[1, 2] / height - 1
    )  # PyBullet/OpenGL convention might differ slightly on sign of this term depending on coordinate system
    # Actually, let's use p.computeProjectionMatrixFOV if we can approximate fov.
    # But we have exact K.
    # Let's try to construct the matrix manually.
    P[10] = -(far + near) / (far - near)
    P[11] = -1
    P[14] = -2 * far * near / (far - near)

    projection_matrix = list(P)

    # Render
    w, h, rgb, depth, seg = p.getCameraImage(
        width,
        height,
        view_matrix,
        projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )

    # Debug: Check if anything was rendered
    # Use depth buffer for mask (OpenGL depth is [0, 1], background is 1.0)
    depth_render = np.array(depth, dtype=np.float32).reshape(h, w)
    mask = depth_render < 0.99  # Use threshold slightly less than 1.0
    print(f"Rendered pixels (depth < 0.99): {np.sum(mask)}")
    print(f"Depth min: {depth_render.min()}, max: {depth_render.max()}")

    # Debug: Project object center to see where it is
    # Object center in OpenGL Camera frame is T_gl
    # Project T_gl using projection_matrix
    # P_gl is column-major 4x4
    P_mat = np.array(projection_matrix).reshape(4, 4).T
    pos_hom = np.array([T_gl[0], T_gl[1], T_gl[2], 1.0])
    pos_clip = P_mat @ pos_hom
    pos_ndc = pos_clip / pos_clip[3]
    print(f"Object center NDC: {pos_ndc[:3]}")
    # NDC is [-1, 1]. If outside, it's clipped.

    if np.sum(mask) == 0:
        print("WARNING: Nothing rendered! Object might be out of view.")

    # Overlay
    rgb_render = np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]

    # Set background to black for visualization
    rgb_render[~mask] = [0, 0, 0]

    # Load target image
    target_img = cv2.imread(target_image_path)
    if target_img is None:
        # If target_image_path was a template, load it with plt/cv2
        target_img = plt.imread(target_image_path)
        if target_img.dtype == np.float32:
            target_img = (target_img * 255).astype(np.uint8)
        target_img = target_img[:, :, :3]
        # Convert RGB to BGR for OpenCV
        target_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2BGR)

    target_img = cv2.resize(target_img, (width, height))

    # Blend
    # Create mask from render (where alpha > 0 or depth < 1.0)
    # PyBullet background depth is 1.0?
    # Let's use alpha from render if available (it is RGBA)
    alpha = np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, 3]
    mask = alpha > 0

    overlay = target_img.copy()
    overlay[mask] = cv2.addWeighted(
        target_img[mask], 0.5, cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)[mask], 0.5, 0
    )

    # Side-by-side comparison
    # target_img is BGR, rgb_render is RGB
    rgb_render_bgr = cv2.cvtColor(rgb_render, cv2.COLOR_RGB2BGR)
    side_by_side = np.hstack((target_img, rgb_render_bgr))
    cv2.imwrite("pose_estimation_side_by_side.png", side_by_side)
    print("Saved pose_estimation_side_by_side.png")

    cv2.imwrite("pose_estimation_result.png", overlay)
    print("Saved pose_estimation_result.png")


def estimate_pose(
    target_image_path,
    template_dir,
    urdf_path,
    sim_log_path=None,
    device="cpu",
    forced_template_path=None,
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = load_model(device, checkpoint_path)

    if target_image_path is None:
        target_image_path = os.path.join(template_dir, "template_000_azim_0.0.png")

    print(f"Target image: {target_image_path}")

    # 1. Find best template
    best_matches = 0
    best_template = None
    best_match_data = None

    if forced_template_path:
        print(f"Using forced template: {forced_template_path}")
        best_template = os.path.basename(forced_template_path)
        # We still need to run inference to get matches
        images = load_images([forced_template_path, target_image_path], size=512)
        matches_template, matches_target, view1, pred1, view2, pred2 = (
            FeatureMatcher.mast3r_inference(images, model, device)
        )
        best_matches = len(matches_template)
        best_match_data = (matches_template, matches_target)
    else:
        template_files = sorted(
            [
                f
                for f in os.listdir(template_dir)
                if f.endswith(".png") and "depth" not in f
            ]
        )

        # Limit to a few templates for speed if needed
        # template_files = template_files[::2]

        # Use all templates or subset as previously defined
        # Restoring full search or keeping user's subset?
        # User had [7:8], let's try to search all if not forced, or maybe just keep the subset if that was intended.
        # But for "reuse best template" to be meaningful, the first run should probably search.
        # I will revert to searching all (or a larger subset) if not forced,
        # assuming the [7:8] was a temporary debug hack by the user or me.
        # Actually, let's just iterate over `template_files` (all of them) to be safe/correct.

        for template_file in template_files:
            template_path = os.path.join(template_dir, template_file)

            images = load_images([template_path, target_image_path], size=512)

            matches_template, matches_target, view1, pred1, view2, pred2 = (
                FeatureMatcher.mast3r_inference(images, model, device)
            )

            num_matches = len(matches_template)
            # print(f"Template {template_file}: {num_matches} matches")

            if num_matches > best_matches:
                best_matches = num_matches
                best_template = template_file
                best_match_data = (matches_template, matches_target)

    print(f"Best match: {best_template} with {best_matches} matches")

    if best_matches < 10:
        print("Not enough matches found.")
        return None, None, None  # Return None to indicate failure

    matches_template, matches_target = best_match_data

    # Get target image shape from the loaded images
    # images is a list of tensors. images[0] is (B, 3, H, W).
    # We passed [template, target]. So it's a batch of 2?
    # load_images returns a list of dicts or tensors?
    # dust3r load_images returns list of dicts usually?
    # No, the log says "Loading a list of 2 images".
    # In estimate_pose.py: images = load_images(...)
    # mast3r_inference takes `images`.
    # Let's look at mast3r_inference signature in feature_matcher_tools.py.
    # It calls `inference([tuple(images)], ...)`
    # `load_images` returns a list of numpy arrays or tensors.
    # Actually, let's just use the shape from `view2` which is returned by mast3r_inference.
    # view2['true_shape'] might be the original shape.
    # view2['img'] is the tensor shape.

    # Let's inspect view2 from the best match
    # _, _, _, _, view2, _ = FeatureMatcher.mast3r_inference(images, model, device)
    # This re-runs inference, which is wasteful but safe since we didn't save view2.
    # Wait, we didn't save view2 in the loop. We only saved matches.
    # We should probably save the shape or re-run.
    # Or just check the shape of the target image we loaded?
    # We loaded `target_img` at the end for visualization.
    # But `load_images` resizes it.
    # The matches are in the resized coordinates.
    # We need the shape of the resized image.

    # Let's just re-run inference for the best match to get everything fresh and correct.
    print("Re-running inference for best match to get metadata...")
    template_path = os.path.join(template_dir, best_template)
    images = load_images([template_path, target_image_path], size=512)
    matches_template, matches_target, view1, pred1, view2, pred2 = (
        FeatureMatcher.mast3r_inference(images, model, device)
    )

    # Target image shape (resized)
    # view2['img'] is (1, 3, H, W)
    h_target, w_target = view2["img"].shape[2], view2["img"].shape[3]
    print(f"Target image resized shape: {w_target}x{h_target}")

    # 2. Lift template points to 3D
    # Load template pose info
    # best_template is like "template_000_azim_0.0.png"
    # pose file is like "template_000_pose.npz"
    template_idx = best_template.split("_")[1]  # "000"
    pose_file = f"template_{template_idx}_pose.npz"
    pose_data = np.load(os.path.join(template_dir, pose_file))

    view_matrix = pose_data["view_matrix"].reshape(4, 4).T

    depth_map = pose_data["depth"]  # (H, W)

    # Camera intrinsics used for generation
    fov = 60
    h_temp, w_temp = depth_map.shape
    K = get_intrinsics(fov, w_temp, h_temp)

    # Scale matches to depth map size
    # matches are in (w_target, h_target) space?
    # No, matches_template are in template image space (resized).
    # template image was resized to 512x384 (or similar).
    # We need to check template resized shape too.
    h_temp_resized, w_temp_resized = view1["img"].shape[2], view1["img"].shape[3]

    scale_x = w_temp / w_temp_resized
    scale_y = h_temp / h_temp_resized

    matches_template_scaled = matches_template.copy().astype(np.float32)
    matches_template_scaled[:, 0] *= scale_x
    matches_template_scaled[:, 1] *= scale_y

    # Unproject
    points_3d, valid_indices = unproject_points(
        matches_template_scaled, depth_map, K, view_matrix
    )

    if len(points_3d) < 4:
        print("Not enough valid 3D points.")
        return

    # Filter target matches
    matches_target_valid = matches_target[valid_indices]

    # Scale target matches to original target image size?
    # load_images resizes to 512.
    # If we want pose in original image, we need to scale back.
    # But for visualization we can just use the resized image or scale K.
    # Let's assume we work in 512x512 space for PnP.
    K_target = get_intrinsics(fov, w_target, h_target)  # Assuming target has same FOV

    # Make contiguous for safety
    points_3d = np.ascontiguousarray(points_3d, dtype=np.float64)

    # # -------- NEW: Ground-plane alignment --------
    # R_ground, points_3d_aligned = align_ground_plane(points_3d)

    # # Apply same rotation to the PnP correspondences (2D stays same, 3D rotates)
    # points_3d = points_3d_aligned

    matches_target_valid = np.ascontiguousarray(matches_target_valid, dtype=np.float64)

    # 3. Solve PnP
    # points_3d: (N, 3) object points
    # matches_target_valid: (N, 2) image points

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d,
        matches_target_valid,
        K_target,
        None,
        iterationsCount=1000,
        reprojectionError=2.0,
        flags=cv2.SOLVEPNP_EPNP,
    )

    if not success:
        print("PnP failed.")
        return

    print(
        f"Estimated Pose (Camera Frame):\nRvec: {rvec.flatten()}\nTvec: {tvec.flatten()}"
    )

    # --- Calculate World Pose ---

    # Try to load camera_pose.json from the same directory as target_image_path
    target_dir = os.path.dirname(target_image_path)
    camera_pose_path = os.path.join(target_dir, "camera_pose.json")

    view_matrix_gl = None

    if os.path.exists(camera_pose_path):
        with open(camera_pose_path, "r") as f:
            camera_params = json.load(f)
        print(f"Loaded camera params from {camera_pose_path}")

        # Use loaded params
        # We saved "view_matrix" (16 floats, column-major)
        view_matrix_gl = np.array(camera_params["view_matrix"]).reshape(4, 4).T

        # We also saved "pos" directly
        cam_pos_world = np.array(camera_params["pos"])
        print(f"Camera Position (World, from JSON): {cam_pos_world}")

    else:
        print(
            "Warning: camera_pose.json not found. Using default hardcoded parameters."
        )
        # Camera parameters from main.py
        cameraDistance = 3.0
        cameraYaw = 225
        cameraPitch = -30
        cameraTargetPosition = [0, 0, 0]

        # Compute View Matrix (World -> Camera_GL)
        view_matrix_gl = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cameraTargetPosition,
            distance=cameraDistance,
            yaw=cameraYaw,
            pitch=cameraPitch,
            roll=0,
            upAxisIndex=2,
        )
        view_matrix_gl = np.array(view_matrix_gl).reshape(4, 4).T

    # Camera_GL -> World
    T_world_gl = np.linalg.inv(view_matrix_gl)

    # GL -> ROS
    # GL: +X right, +Y up, +Z back
    # ROS: +X forward, +Y left, +Z up
    # X_ros = -Z_gl
    # Y_ros = -X_gl
    # Z_ros = Y_gl
    # Matrix columns are images of GL basis vectors in ROS frame?
    # No, we want T_gl_ros (ROS to GL) to convert T_world_gl to T_world_ros
    # T_world_ros = T_world_gl @ T_gl_ros

    # ROS -> GL
    # X_ros (fwd) -> -Z_gl
    # Y_ros (left) -> -X_gl
    # Z_ros (up) -> Y_gl
    R_ros_to_gl = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    T_gl_ros = np.eye(4)
    T_gl_ros[:3, :3] = R_ros_to_gl

    T_world_camera = T_world_gl @ T_gl_ros

    if not os.path.exists(camera_pose_path):
        print(f"Camera Position (World, calculated): {T_world_camera[:3, 3]}")

    # Car -> Camera_CV (from PnP)
    R_cv, _ = cv2.Rodrigues(rvec)
    # Camera (OpenCV) -> Marker
    # rvecs[i] is the rotation vector from solvePnP
    t_cv = tvec.flatten()

    # Construct 4x4 matrix for Car in OpenCV Camera Frame
    T_cv_car = np.eye(4)
    T_cv_car[:3, :3] = R_cv
    T_cv_car[:3, 3] = t_cv

    # Convert from OpenCV Camera Frame (Z-forward, X-right, Y-down)
    # to ROS Camera Frame (X-forward, Y-left, Z-up)
    # Rotation: X_ros = Z_cv, Y_ros = -X_cv, Z_ros = -Y_cv
    # This corresponds to a rotation matrix:
    # [[0, 0, 1],
    #  [-1, 0, 0],
    #  [0, -1, 0]]

    R_cv_to_ros = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]  # correct axes conversion

    # T_ros_car = R_cv_to_ros * T_cv_car
    # We need to apply this rotation to the coordinate system.
    # The position in ROS frame: P_ros = R_cv_to_ros * P_cv
    # The orientation in ROS frame: R_ros = R_cv_to_ros * R_cv

    T_camera_car = np.eye(4)
    T_camera_car[:3, :3] = R_cv_to_ros @ T_cv_car[:3, :3]
    T_camera_car[:3, 3] = R_cv_to_ros @ T_cv_car[:3, 3]

    # Calculate Car pose in World frame
    # T_world_car = T_world_cam * T_camera_car
    T_world_car = T_world_camera @ T_camera_car

    print("\nEstimated Pose (World Frame):")
    estimated_pos = T_world_car[:3, 3]
    print(f"Translation: {estimated_pos}")
    # Convert rotation to Euler for easier reading
    rot_matrix = T_world_car[:3, :3]
    estimated_rot = Rotation.from_matrix(rot_matrix).as_quat()
    # We can use scipy or just print matrix
    print(f"Rotation Matrix:\n{rot_matrix}")

    # Wall is at [0, 0, 0] identity. So T_car_wall is same as T_car_world.
    print(f"Distance to Wall (Origin): {np.linalg.norm(estimated_pos)}")

    # Calculate Euler angles (Yaw, Pitch, Roll)
    # PyBullet uses Z-up, so Yaw is rotation around Z.
    # Scipy 'zyx' corresponds to intrinsic rotations: Z (yaw), then Y (pitch), then X (roll).
    # Or extrinsic 'xyz'?
    # Let's use 'zyx' which is standard for Yaw-Pitch-Roll.
    # The first angle returned will be rotation around Z.

    r = Rotation.from_matrix(rot_matrix)
    yaw, pitch, roll = r.as_euler("zyx", degrees=True)
    print(f"Estimated Yaw (around World Z): {yaw:.2f} degrees")

    # --- Calculate Error if Sim Log Provided ---
    if sim_log_path and os.path.exists(sim_log_path):
        import csv

        try:
            # Extract frame index from filename (e.g., "81.png" -> 81)
            filename = os.path.basename(target_image_path)
            frame_idx = int(os.path.splitext(filename)[0])

            with open(sim_log_path, "r") as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                rows = list(reader)

                if frame_idx < len(rows):
                    row = rows[frame_idx]
                    # px, py, pz are indices 1, 2, 3
                    gt_pos = np.array([float(row[1]), float(row[2]), float(row[3])])
                    # rx, ry, rz, rw are indices 4, 5, 6, 7
                    gt_rot = np.array(
                        [float(row[4]), float(row[5]), float(row[6]), float(row[7])]
                    )

                    # Convert GT quaternion to Yaw
                    # PyBullet quaternions are [x, y, z, w]
                    # Scipy expects [x, y, z, w]
                    r_gt = Rotation.from_quat(gt_rot)
                    gt_yaw, gt_pitch, gt_roll = r_gt.as_euler("zyx", degrees=True)

                    error_vec = estimated_pos - gt_pos
                    error_dist = np.linalg.norm(error_vec)
                    gt_dist = np.linalg.norm(gt_pos)

                    percent_error = (
                        (error_dist / gt_dist) * 100 if gt_dist > 1e-6 else float("inf")
                    )

                    yaw_error = abs(yaw - gt_yaw)
                    # Handle wrap around (e.g. 179 vs -179 is 2 deg error, not 358)
                    if yaw_error > 180:
                        yaw_error = 360 - yaw_error

                    print("\n--- Verification ---")
                    print(f"Frame: {frame_idx}")
                    print(f"Ground Truth Position: {gt_pos}")
                    print(f"Estimated Position:    {estimated_pos}")
                    print(f"Ground Truth Rotation: {gt_rot}")
                    print(f"Estimated Rotation:    {estimated_rot}")
                    print(f"Ground Truth Yaw:      {gt_yaw:.2f} degrees")
                    print(f"Estimated Yaw:         {yaw:.2f} degrees")
                    print(f"Absolute Error: {error_dist:.4f} m")
                    print(f"Percent Error:  {percent_error:.2f}%")
                    print(f"Yaw Error:      {yaw_error:.2f} degrees")
                else:
                    print(
                        f"\nWarning: Frame {frame_idx} not found in sim log (log has {len(rows)} rows)."
                    )
        except Exception as e:
            print(f"\nError calculating error from sim log: {e}")

    # Visualize Matches
    # matches_template_scaled: (N, 2) in template image
    # matches_target_valid: (N, 2) in target image
    # We need to load the images to draw on them
    # template_path is defined in the loop, we need to reload it or keep it
    template_path = os.path.join(template_dir, best_template)
    img1 = cv2.imread(template_path)
    img2 = cv2.imread(target_image_path)

    # Resize img1 to match the scaling we did?
    # matches_template_scaled was scaled to depth map size (256x256)
    # But we want to visualize on original or resized image.
    # Let's visualize on 512x512 (which mast3r used)
    # We need to scale points back to 512x512 for visualization
    # scale_x = w_temp / w_temp_resized (256 / 512 = 0.5)
    # So divide by scale to get back to 512

    # Actually, let's just use the matches from mast3r directly (matches_template, matches_target)
    # They are in 512x512 space (or whatever mast3r output)
    # matches_template and matches_target are the raw outputs from mast3r_inference
    # They correspond to the images passed to mast3r_inference
    # which were resized to 512.

    # So we should resize images to 512 for visualization
    img1 = cv2.resize(img1, (512, 384))  # Aspect ratio?
    img2 = cv2.resize(img2, (512, 384))

    # Draw matches
    # matches_template and matches_target are (N, 2)
    # valid_indices filters them

    kp1 = [
        cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1)
        for p in matches_template[valid_indices]
    ]
    kp2 = [
        cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1)
        for p in matches_target[valid_indices]
    ]

    matches_draw = [
        cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(kp1))
    ]

    match_img = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite("matches_visualization.png", match_img)
    print("Saved matches_visualization.png")

    # 4. Visualize
    return tvec.flatten(), yaw, best_template


if __name__ == "__main__":
    template_dir = "templates"
    urdf_path = "my_racecar/my_racecar.urdf"
    # Use an image from the latest run where camera_pose.json was generated

    RUN = "1764808669"
    FRAME_1 = 56
    FRAME_2 = 58  # 3 frames difference
    DT = 1.0 / 240.0

    # Calculate average yaw between FRAME_1 and FRAME_2
    yaws = []
    last_tvec = None
    speeds = []

    for frame_idx in range(FRAME_1, FRAME_2 + 1):
        tvec, yaw, _ = estimate_pose(
            f"sim_gifs/run_{RUN}/{frame_idx}.png",
            template_dir,
            urdf_path,
            sim_log_path=f"sim_logs/run_{RUN}.csv",
        )
        yaws.append(yaw)

        if last_tvec is None:
            last_tvec = tvec
        else:
            displacement = np.linalg.norm(tvec - last_tvec)
            speeds.append(displacement / DT)
            last_tvec = tvec

    avg_yaw = sum(yaws) / len(yaws)
    avg_speed = sum(speeds) / len(speeds)

    print("\n=== Speed Estimation ===")
    print(f"Frame {FRAME_1} -> {FRAME_2} (dt={DT:.4f}s)")
    print(f"Average Yaw: {avg_yaw:.2f} degrees")
    print(f"Average Speed: {avg_speed:.4f} m/s")
