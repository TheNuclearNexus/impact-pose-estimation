import pybullet as p
import pybullet_data
import time
import random
import math
import pygame

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

plane_id = p.loadURDF("plane.urdf")

# --- Cube setup ---
cube_half = [0.1, 0.1, 0.1]
cube_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_half)
cube_red = p.createVisualShape(p.GEOM_BOX, halfExtents=cube_half, rgbaColor=[1, 0, 0, 1])
cube_blue = p.createVisualShape(p.GEOM_BOX, halfExtents=cube_half, rgbaColor=[0, 0, 1, 1])

cube1 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape, baseVisualShapeIndex=cube_red, basePosition=[-1, 0, 0.1])
cube2 = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=cube_shape, baseVisualShapeIndex=cube_blue, basePosition=[1, 0, 0.1])

for cube in (cube1, cube2):
    p.changeDynamics(cube, -1, restitution=0.9, lateralFriction=0.05, linearDamping=0.01, angularDamping=0.01)

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

def reset_cubes():
    # Random symmetrical X positions and small Y jitter
    x1 = random.uniform(-2.0, -1.2)
    x2 = random.uniform(1.2, 2.0)
    y1 = random.uniform(-0.8, 0.8)
    y2 = random.uniform(-0.8, 0.8)
    z = 0.1

    pos1 = [x1, y1, z]
    pos2 = [x2, y2, z]
    orn = [0, 0, 0, 1]  # identity quaternion

    # place cubes
    p.resetBasePositionAndOrientation(cube1, pos1, orn)
    p.resetBasePositionAndOrientation(cube2, pos2, orn)

    # zero angular velocity and set linear velocity toward each other
    speed = random.uniform(6.0, 8.0)  # randomize speed a bit
    v1, v2 = compute_pair_velocities(pos1, pos2, speed)

    p.resetBaseVelocity(cube1, linearVelocity=v1, angularVelocity=[0, 0, 0])
    p.resetBaseVelocity(cube2, linearVelocity=v2, angularVelocity=[0, 0, 0])

# initial run
reset_cubes()
last_collision = False
print("Press R to randomize positions. Collisions play sound.")

# Main loop
while True:
    p.stepSimulation()
    time.sleep(1/240)

    # Detect collisions between cubes
    contacts = p.getContactPoints(cube1, cube2)
    collision_now = len(contacts) > 0

    # Play sound on *new* collision start
    if collision_now and not last_collision:
        impact.play()

    last_collision = collision_now

    # Check for R key
    keys = p.getKeyboardEvents()
    if (114 in keys and keys[114] & p.KEY_WAS_TRIGGERED) or (82 in keys and keys[82] & p.KEY_WAS_TRIGGERED):
        reset_cubes()