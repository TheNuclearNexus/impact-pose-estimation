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
p.setTimeStep(1.0 / 1000.0)

# --- Camera Setup ---

p.resetDebugVisualizerCamera(
    cameraDistance=3.0,
    cameraYaw=225,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

# --- Floor setup ---

plane_id = p.loadURDF("plane.urdf")
floor_texture_id = p.loadTexture("asphalt-texture.png")
p.changeVisualShape(plane_id, -1, textureUniqueId=floor_texture_id)

# --- Wall Setup ---

wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 1, 1])
wall_body = p.createMultiBody(
    baseMass=0,
    baseCollisionShapeIndex=wall,
    basePosition=[0, 0, 0]
)

# --- Cube setup ---
cube_half = [0.1, 0.1, 0.1]
cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_half)
cube_red = p.createVisualShape(p.GEOM_BOX, halfExtents=cube_half, rgbaColor=[1, 0, 0, 1])
cube_blue = p.createVisualShape(p.GEOM_BOX, halfExtents=cube_half, rgbaColor=[0, 0, 1, 1])

#cube1 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape, baseVisualShapeIndex=cube_red, basePosition=[-1, 0, 0.1])
#cube2 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape, baseVisualShapeIndex=cube_blue, basePosition=[1, 0, 0.1])

# -- Car setup ---
import pybullet_data
print(pybullet_data.getDataPath())

car1 = p.loadURDF("my_racecar/my_racecar.urdf", basePosition=[-2, 0, 0.2])
#car2 = p.loadURDF("my_racecar/my_racecar.urdf", basePosition=[2, 0, 0.2])

#for car in (car1, car2):
p.changeDynamics(car1, -1, restitution=0.9, lateralFriction=0.05, linearDamping=0.01, angularDamping=0.01)

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

def reset_cars():
    # Random symmetrical X positions and small Y jitter
    x1 = random.uniform(-2.0, -1.2)
    x2 = random.uniform(1.2, 2.0)
    y1 = random.uniform(-0.8, 0.8)
    y2 = random.uniform(-0.8, 0.8)
    z = 0

    pos1 = [x1, y1, z]
    pos2 = [x2, y2, z]
    orn = [0, 0, 0, 1]  # identity quaternion

    # place cubes
    p.resetBasePositionAndOrientation(car1, pos1, orn)
    #p.resetBasePositionAndOrientation(car2, pos2, orn)

    # zero angular velocity and set linear velocity toward each other
    speed = random.uniform(20.0, 30.0)  # randomize speed a bit
    v1, v2 = compute_pair_velocities(pos1, [0, 0, 0], speed)

    p.resetBaseVelocity(car1, linearVelocity=v1, angularVelocity=[0, 0, 0])
    #p.resetBaseVelocity(car2, linearVelocity=v2, angularVelocity=[0, 0, 0])

# initial run
reset_cars()
last_collision = False
print("Press R to randomize positions. Collisions play sound.")

data_log = []
collision_start_time = None
sim_start_time = time.time()
has_logged = False

def save_sim_data(rows):
    os.makedirs("sim_logs", exist_ok=True)
    filename = f"sim_logs/run_{int(time.time())}.csv"
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time",
            "px","py","pz",
            "ox","oy","oz","ow",
            "vx","vy","vz",
            "wx","wy","wz",
            "collision"
        ])
        w.writerows(rows)
    print("Saved:", filename)

# Main loop
while True:
    p.stepSimulation()
    frame_time = time.time() - sim_start_time
    time.sleep(1/240)

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
    data_log.append([
        frame_time,
        pos[0], pos[1], pos[2],
        orn[0], orn[1], orn[2], orn[3],
        lin_vel[0], lin_vel[1], lin_vel[2],
        ang_vel[0], ang_vel[1], ang_vel[2],
        int(collision_now)
    ])

    last_collision = collision_now

    # After 1 second post-impact, dump to disk
    if collision_start_time is not None and frame_time > collision_start_time + 1.0 and not has_logged:
        save_sim_data(data_log)
        has_logged = True

    # Check for R key
    keys = p.getKeyboardEvents()
    if (114 in keys and keys[114] & p.KEY_WAS_TRIGGERED) or (82 in keys and keys[82] & p.KEY_WAS_TRIGGERED):
        reset_cars()

        # Reset logging parameters
        data_log = []
        collision_start_time = None
        sim_start_time = time.time()
        has_logged = False
