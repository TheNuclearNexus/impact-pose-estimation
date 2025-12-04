from camera_config import FAR
from camera_config import NEAR
from camera_config import FOV
import json
from pathlib import Path
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
import math
import pygame
import csv
import os

# --- Audio setup ---
pygame.mixer.init()
# Replace this path with your own short .wav file
# (e.g. "impact.wav" or "click.wav" in the same directory)
IMPACT_SOUND = "impact.wav"
impact = pygame.mixer.Sound(IMPACT_SOUND)

# --- PyBullet setup ---
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setPhysicsEngineParameter(numSolverIterations=50)
p.setTimeStep(1.0 / 240.0)

# --- Camera Setup ---

p.resetDebugVisualizerCamera(
    cameraDistance=3.0, cameraYaw=225, cameraPitch=-30, cameraTargetPosition=[0, 0, 0]
)

# --- Floor setup ---

plane_id = p.loadURDF("plane.urdf")
floor_texture_id = p.loadTexture("asphalt-texture.png")
p.changeVisualShape(plane_id, -1, textureUniqueId=floor_texture_id)

# --- Wall Setup ---

wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 1, 1])
wall_body = p.createMultiBody(
    baseMass=0, baseCollisionShapeIndex=wall, basePosition=[0, 0, 0]
)
p.changeVisualShape(wall_body, -1, rgbaColor=[0, 1, 0, 1])

# --- Cube setup ---
# cube_half = [0.1, 0.1, 0.1]
# cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_half)
# cube_red = p.createVisualShape(
#     p.GEOM_BOX, halfExtents=cube_half, rgbaColor=[1, 1, 0, 1]
# )
# cube_blue = p.createVisualShape(
#     p.GEOM_BOX, halfExtents=cube_half, rgbaColor=[0, 0, 1, 1]
# )

# cube1 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape, baseVisualShapeIndex=cube_red, basePosition=[-1, 0, 0.1])
# cube2 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape, baseVisualShapeIndex=cube_blue, basePosition=[1, 0, 0.1])

# -- Car setup ---
import pybullet_data

print(pybullet_data.getDataPath())

car1 = p.loadURDF("my_racecar/my_racecar.urdf", basePosition=[-2, 0, 0.2])
# car2 = p.loadURDF("my_racecar/my_racecar.urdf", basePosition=[2, 0, 0.2])

# for car in (car1, car2):
p.changeDynamics(
    car1,
    -1,
    restitution=0.9,
    lateralFriction=0.05,
    linearDamping=0.01,
    angularDamping=0.01,
)


# --- Utility functions ---
def compute_pair_velocities(pos_a, pos_b, speed_mag):
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    dist_xy = math.hypot(dx, dy)
    if dist_xy < 1e-6:
        return ([speed_mag, 0, 0], [-speed_mag, 0, 0])
    ux, uy = dx / dist_xy, dy / dist_xy
    va = [ux * speed_mag, uy * speed_mag, 0]
    vb = [-ux * speed_mag, -uy * speed_mag, 0]
    return va, vb


def quat_from_yaw(yaw):
    """Return a quaternion (x,y,z,w) for rotation about Z by yaw radians."""
    half = yaw / 2
    return [0, 0, math.sin(half), math.cos(half)]


def reset_cars_randomized():
    # Random symmetrical X positions and small Y jitter
    x1 = random.uniform(-3.0, -2)
    y1 = random.uniform(-0.8, 0.8)

    y1 += 0.1 * -1 if y1 < 0 else 1

    z = 0

    pos1 = [x1, y1, z]
    # zero angular velocity and set linear velocity toward each other
    speed = random.uniform(10.0, 13.0)  # randomize speed a bit
    v1, v2 = compute_pair_velocities(pos1, [0, 0, 0], speed)

    # Compute facing yaw angles for each cube based on velocity
    yaw1 = math.atan2(-v1[1], -v1[0])
    orn1 = quat_from_yaw(yaw1)

    # place cubes
    p.resetBasePositionAndOrientation(car1, pos1, orn1)
    p.resetBaseVelocity(car1, linearVelocity=v1, angularVelocity=[0, 0, 0])
    # p.resetBaseVelocity(car2, linearVelocity=v2, angularVelocity=[0, 0, 0])

    global last_positions
    last_positions = {
        car1: pos1,
    }

def reset_cars(x_pos, y_pos, yaw, speed):
    # Random symmetrical X positions and small Y jitter
    #x1 = random.uniform(-3.0, -2)
    #y1 = random.uniform(-0.8, 0.8)

    y_pos += 0.1 * -1 if y_pos < 0 else 1

    z = 0

    pos1 = [x_pos, y_pos, z]
    # zero angular velocity and set linear velocity toward each other
    v1, v2 = compute_pair_velocities(pos1, [0, 0, 0], speed)

    orn1 = quat_from_yaw(yaw)

    # place cubes
    p.resetBasePositionAndOrientation(car1, pos1, orn1)
    p.resetBaseVelocity(car1, linearVelocity=v1, angularVelocity=[0, 0, 0])
    # p.resetBaseVelocity(car2, linearVelocity=v2, angularVelocity=[0, 0, 0])

    global last_positions
    last_positions = {
        car1: pos1,
    }

last_positions = {}

def save_sim_data(rows):
    os.makedirs("sim_logs", exist_ok=True)
    filename = f"sim_logs/run_{int(time.time())}.csv"
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "time",
                "px",
                "py",
                "pz",
                "ox",
                "oy",
                "oz",
                "ow",
                "vx",
                "vy",
                "vz",
                "wx",
                "wy",
                "wz",
                "collision",
            ]
        )
        w.writerows(rows)
    print("Saved:", filename)

    return filename


WIDTH = 512
HEIGHT = 384


def save_sim_frames(frames):
    from PIL import Image

    os.makedirs(f"sim_gifs/run_{int(time.time())}", exist_ok=True)
    base = Path(f"sim_gifs/run_{int(time.time())}")
    for i, frame in enumerate(frames):
        img = Image.frombytes("RGBA", (WIDTH, HEIGHT), (frame * 255).astype(np.uint8))
        img.save(base / f"{i}.png")

        print("Saved:", base / f"{i}.png")

    # Save camera pose
    cam = p.getDebugVisualizerCamera()
    # cam[2] is the view matrix (tuple of 16 floats, column-major)
    view_matrix = np.array(cam[2]).reshape(4, 4).T
    # Inverse view matrix gives Camera -> World transform
    inv_view_matrix = np.linalg.inv(view_matrix)
    # Camera position is the translation part of the inverse view matrix
    camera_pos = inv_view_matrix[:3, 3]

    with open(base / "camera_pose.json", "w") as f:
        json.dump(
            {
                "pos": camera_pos.tolist(),
                "view_matrix": cam[2],
                "projection_matrix": cam[3],
                "yaw": cam[8],
                "pitch": cam[9],
                "dist": cam[10],
                "target": cam[11],
            },
            f,
        )


projection_matrix = p.computeProjectionMatrixFOV(
    fov=FOV, aspect=WIDTH / HEIGHT, nearVal=NEAR, farVal=FAR
)

def run_simulation(should_use_randomized, x_pos=0, y_pos=0, yaw=0, speed=10):

    # initial run
    last_collision = False
    if (should_use_randomized):
        reset_cars_randomized()
    else:
        reset_cars(x_pos, y_pos, yaw, speed)
    print("Press R to randomize positions. Collisions play sound.")

    data_log = []
    sim_frames = []
    collision_start_time = None
    sim_start_time = time.time()
    has_logged = False

    # Main loop
    while True:
        p.stepSimulation()
        frame_time = time.time() - sim_start_time
        time.sleep(1 / 240)

        if not has_logged:
            # Record gif
            frame = p.getCameraImage(
                WIDTH,
                HEIGHT,
                projectionMatrix=projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
            print(len(frame[2]))
            sim_frames.append(np.reshape(frame[2], (HEIGHT, WIDTH, 4)) * 1.0 / 255.0)

        # Detect collisions between cubes
        contacts = p.getContactPoints(car1, wall_body)
        collision_now = len(contacts) > 0

        # Get car state
        pos, orn = p.getBasePositionAndOrientation(car1)
        lin_vel, ang_vel = p.getBaseVelocity(car1)

        # Play sound on *new* collision start
        if collision_now and not last_collision:
            if collision_start_time is None:
                collision_start_time = frame_time
            impact.play()

        # Store one row of data per frame
        data_log.append(
            [
                frame_time,
                pos[0],
                pos[1],
                pos[2],
                orn[0],
                orn[1],
                orn[2],
                orn[3],
                lin_vel[0],
                lin_vel[1],
                lin_vel[2],
                ang_vel[0],
                ang_vel[1],
                ang_vel[2],
                int(collision_now),
            ]
        )

        last_collision = collision_now

        filename = None

        # After 1 second post-impact, dump to disk
        if (
            collision_start_time is not None
            and frame_time > collision_start_time + 1.0
            and not has_logged
        ):
            filename = save_sim_data(data_log)
            save_sim_frames(sim_frames)
            has_logged = True
            return filename

        # Check for R key
        keys = p.getKeyboardEvents()
        if (114 in keys and keys[114] & p.KEY_WAS_TRIGGERED) or (
            82 in keys and keys[82] & p.KEY_WAS_TRIGGERED
        ):
            if (should_use_randomized):
                reset_cars_randomized()
            else:
                reset_cars(x_pos=x_pos, y_pos=y_pos, yaw=yaw, speed=speed)

            # Reset logging parameters
            data_log = []
            sim_frames = []
            collision_start_time = None
            sim_start_time = time.time()
            has_logged = False

filename1 = run_simulation(should_use_randomized=True)
filename2 = run_simulation(should_use_randomized=False, x_pos=-2.5, y_pos=0, yaw=-180, speed=10)

from compare_sims import compare_runs

compare_runs(filename1, filename2)
