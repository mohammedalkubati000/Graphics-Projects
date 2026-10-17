import math
import numpy as np

WORLD_LEFT , WORLD_RIGHT = -5.5 , 5.5
WORLD_BOTTOM , WORLD_TOP = -4.0 , 4.0

def T(x, y):
    return np.array([
        [1, 0, x],
        [0, 1, y],
        [0, 0, 1]
    ], dtype=np.float32)


def S(sx, sy=None):
    if sy is None:
        sy = sx
    return np.array([
        [sx,  0, 0],
        [0,  sy, 0],
        [0,   0, 1]
    ], dtype=np.float32)

def R(angle_degrees):
    a = math.radians(angle_degrees)
    c , s = math.cos(a) , math.sin(a)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)

def transform_point(M, x, y):
    p = np.array([x, y, 1], dtype=np.float32)
    result = M @ p
    return result[0], result[1]

def rectangle(length, width):
    x , y = length / 2.0 , width / 2.0
    return np.array([
        [-x, -y],
        [ x, -y],
        [ x,  y],
        [-x,  y]
    ], dtype=np.float32)

def centered_rectangle(length, width):
    return rectangle(length, width)

def transform_polygon(M, polygon):
    result = []
    for x, y in polygon:
        tx, ty = transform_point(M, x, y)
        result.append([tx, ty])
    return np.array(result, dtype=np.float32)

def ortho_matrix():
    """Orthographic projection matrix P mapping world coords to NDC."""
    sx = 2.0 / (WORLD_RIGHT - WORLD_LEFT)
    sy = 2.0 / (WORLD_TOP - WORLD_BOTTOM)
    tx = -(WORLD_RIGHT + WORLD_LEFT) / (WORLD_RIGHT - WORLD_LEFT)
    ty = -(WORLD_TOP + WORLD_BOTTOM) / (WORLD_TOP - WORLD_BOTTOM)
    return np.array([
        [sx,  0.0, tx],
        [0.0, sy,  ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

""""""