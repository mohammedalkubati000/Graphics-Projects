import math
import numpy as np
from mat4 import look_at, perspective, ortho


class OrbitCamera:
    def __init__(self, target=(0.0, 0.0, 0.0), yaw=35.0, pitch=28.0, radius=55.0):
        self.home = (tuple(target), yaw, pitch, radius)
        self.reset()

    def reset(self):
        target, yaw, pitch, radius = self.home
        self.target = np.array(target, dtype=np.float32)
        self.yaw, self.pitch, self.radius = yaw, pitch, radius   # degrees, degrees, units
        self.fov = 50.0                 # vertical field of view, degrees
        self.near, self.far = 0.5, 300.0
        self.orthographic = False

    # ---------------------------------------------------------------- input
    def orbit(self, d_yaw, d_pitch):
        self.yaw = (self.yaw + d_yaw) % 360.0
        # Clamp pitch below 90 deg: at exactly 90 the gaze is parallel to
        # up = (0,1,0), cross(up, w) = 0 and look_at cannot build u.
        #
        # STAGE 1 EXERCISE: comment out the clamp below, drag the mouse
        # past the pole, and watch look_at raise / produce NaN.
        self.pitch = max(-89.0, min(89.0, self.pitch + d_pitch))
       # self.pitch = self.pitch + d_pitch        # <-- UNCOMMENT TO BREAK

    def zoom(self, factor):
        self.radius = max(2.0, min(250.0, self.radius * factor))

    # ------------------------------------------------------------- matrices
    def eye(self):
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        offset = np.array([math.cos(p) * math.sin(y),        # spherical -> cartesian
                           math.sin(p),
                           math.cos(p) * math.cos(y)], dtype=np.float32)
        return self.target + self.radius * offset

    def view(self):
        return look_at(self.eye(), self.target, (0.0, 1.0, 0.0))

    def projection(self, aspect):
        if self.orthographic:
            # Choose the box so the TARGET looks the same size in both modes:
            # at distance r a perspective camera sees a half-height of r*tan(fov/2).
            half_h = self.radius * math.tan(math.radians(self.fov) / 2.0)
            half_w = half_h * aspect
            return ortho(-half_w, half_w, -half_h, half_h, self.near, self.far)
        return perspective(self.fov, aspect, self.near, self.far)

    def describe(self):
        mode = "ORTHO" if self.orthographic else "persp"
        return f"{mode} fov={self.fov:.0f} near={self.near:g} far={self.far:g} r={self.radius:.0f}"