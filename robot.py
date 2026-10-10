import numpy as np
from utils import T, R, rectangle

L1 = 2.4
L2 = 2.0
theta1 = 0.0
theta2 = 0.0
theta3 = 0.0
gripper_opening = 1.0
MAX_GRIPPER_GAP = 0.75

def transform_polygon(pts, M):
    transformed = []
    for p in pts:
        v = np.array([p[0], p[1], 1.0])
        vt = M @ v
        transformed.append((vt[0], vt[1]))
    return transformed

def get_local_matrices():
    # Exercise 2: Rest Pose Assembly using translations along x-axis
    Mshoulder = T(0, 0)
    Mupper    = Mshoulder @ T(0, 0)
    Mfore     = Mupper @ T(L1, 0)
    Mhand     = Mfore @ T(L2, 0)
    return Mshoulder, Mupper, Mfore, Mhand

def draw_shoulder(renderer, M=None):
    if M is None:
        M, _, _, _ = get_local_matrices()
    housing = transform_polygon(rectangle(0.75, 1.00), M @ T(0, 0.15))
    renderer.polygon(housing, (0.65, 0.68, 0.72))
    
    c1 = M @ np.array([0, 0, 1.0])
    renderer.circle(c1[0], c1[1], 0.42, (0.12, 0.13, 0.15))
    renderer.circle(c1[0], c1[1], 0.20, (0.85, 0.72, 0.15))

def draw_upper_arm(renderer, M=None):
    if M is None:
        _, M, _, _ = get_local_matrices()
    arm = transform_polygon(rectangle(2.40, 0.48), M @ T(1.20, 0))
    renderer.polygon(arm, (0.75, 0.42, 0.12))
    strip = transform_polygon(rectangle(1.97, 0.12), M @ T(1.20, 0))
    renderer.polygon(strip, (0.20, 0.22, 0.25))

def draw_elbow(renderer, M=None):
    if M is None:
        _, _, M, _ = get_local_matrices()
    c1 = M @ np.array([0, 0, 1.0])
    renderer.circle(c1[0], c1[1], 0.38, (0.10, 0.11, 0.12))
    renderer.circle(c1[0], c1[1], 0.19, (0.85, 0.72, 0.15))

def draw_forearm(renderer, M=None):
    if M is None:
        _, _, M, _ = get_local_matrices()
    arm = transform_polygon(rectangle(2.00, 0.40), M @ T(1.00, 0))
    renderer.polygon(arm, (0.72, 0.40, 0.10))
    strip = transform_polygon(rectangle(1.60, 0.10), M @ T(1.00, 0))
    renderer.polygon(strip, (0.20, 0.22, 0.25))

def draw_wrist(renderer, M=None):
    if M is None:
        _, _, _, M = get_local_matrices()
    c1 = M @ np.array([0, 0, 1.0])
    renderer.circle(c1[0], c1[1], 0.30, (0.10, 0.11, 0.12))
    renderer.circle(c1[0], c1[1], 0.14, (0.85, 0.72, 0.15))

def draw_gripper(renderer, M=None):
    if M is None:
        _, _, _, M = get_local_matrices()
    gap = gripper_opening * MAX_GRIPPER_GAP
    palm = transform_polygon(rectangle(0.65, 0.55), M @ T(0.28, 0))
    renderer.polygon(palm, (0.48, 0.50, 0.54))
    finger_base = rectangle(0.65, 0.16)
    upper_finger = transform_polygon(finger_base, M @ T(0.72, gap / 2.0))
    lower_finger = transform_polygon(finger_base, M @ T(0.72, -gap / 2.0))
    renderer.polygon(upper_finger, (0.60, 0.62, 0.66))
    renderer.polygon(lower_finger, (0.60, 0.62, 0.66))

def draw_robot(renderer):
    Mshoulder, Mupper, Mfore, Mhand = get_local_matrices()
    draw_shoulder(renderer, Mshoulder)
    draw_upper_arm(renderer, Mupper)
    draw_elbow(renderer, Mfore)
    draw_forearm(renderer, Mfore)
    draw_wrist(renderer, Mhand)
    draw_gripper(renderer, Mhand)