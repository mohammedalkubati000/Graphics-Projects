import numpy as np
from utils import T, S, R

class Camera2D:
    def __init__(self):
        self.position = np.array([0.0, 0.0], dtype=np.float32)
        self.zoom = 1.0
        self.rotation = 0.0

    def view_matrix(self):
        return S(1.0 / self.zoom) @ R(-self.rotation) @ T(-self.position[0], -self.position[1])
