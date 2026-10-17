import math
import numpy as np


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError("cannot normalize a zero-length vector")
    return v / n


def identity():
    return np.eye(4, dtype=np.float32)


# ------------------------------------------------------- affine transforms
def translate(tx, ty, tz):
    M = identity()
    M[0, 3], M[1, 3], M[2, 3] = tx, ty, tz
    return M


def scale(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0]).astype(np.float32)


def rotate_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def rotate_y(deg):
    # z turns toward x, which is why -sin sits in the bottom-left
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def rotate_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)


def basis_from_w(w):
    """Orthonormal right-handed (u, v, w) with w = the given direction."""
    w = normalize(w)
    t = w.copy()
    t[np.argmin(np.abs(t))] = 1.0          # any vector not parallel to w
    u = normalize(np.cross(t, w))
    v = np.cross(w, u)
    return u, v, w


def rotate_axis(axis, deg):
    """Rotation about an arbitrary axis through the origin:  B . Rz . B^T"""
    u, v, w = basis_from_w(axis)
    B = identity()
    B[:3, 0], B[:3, 1], B[:3, 2] = u, v, w    # columns = the new frame
    return B @ rotate_z(deg) @ B.T


# -------------------------------------------------------------- normals
def normal_matrix(M):
    """Inverse transpose of the upper-left 3x3.  Renormalise after use."""
    return np.linalg.inv(M[:3, :3]).T.astype(np.float32)


# --------------------------------------------------------------- viewing
def look_at(eye, target, up):
    """View matrix  M_cam = [u v w e]^-1 = R . T(-e)."""
    eye = np.asarray(eye, dtype=np.float32)
    g = np.asarray(target, dtype=np.float32) - eye
    w = -normalize(g)                          # camera looks along -w
    u = normalize(np.cross(up, w))             # fails if up is parallel to g
    v = np.cross(w, u)
    R = identity()
    R[0, :3], R[1, :3], R[2, :3] = u, v, w     # ROWS: transpose = inverse
    return R @ translate(*(-eye))


def ortho(left, right, bottom, top, near, far):
    """OpenGL-style: maps the box to [-1,1]^3, near -> -1, far -> +1."""
    M = identity()
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (far - near)
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(far + near) / (far - near)
    return M


def perspective(fovy_deg, aspect, near, far):
    """OpenGL-style perspective.  near, far are POSITIVE distances."""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (far + near) / (near - far)
    M[2, 3] = 2.0 * far * near / (near - far)
    M[3, 2] = -1.0                             # w_clip = -z_eye
    return M


# ------------------------------------------------------ applying matrices
def transform_dir(M, d):
    """Apply M to a direction (w = 0): translation has no effect."""
    return (M @ np.array([d[0], d[1], d[2], 0.0], dtype=np.float32))[:3]


def to_clip(M, p):
    """Apply M to a point and return the 4D clip coordinates (no divide)."""
    return M @ np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)


def gl_bytes(M):
    """Row-major NumPy -> column-major bytes for a GLSL mat3/mat4 uniform."""
    return np.ascontiguousarray(np.asarray(M, dtype=np.float32).T).tobytes()