import math
import moderngl
import numpy as np
from utils import ortho_matrix

VERTEX_SHADER = """
#version 330
in vec2 in_position;
uniform mat3 u_model;
uniform mat3 u_projection;
void main()
{
    vec3 p = u_model * vec3(in_position, 1.0);
    vec3 ndc = u_projection * p;
    gl_Position = vec4(ndc.xy, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform vec3 color;
out vec4 fragColor;
void main()
{
    fragColor = vec4(color, 1.0);
}
"""

class Renderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=VERTEX_SHADER,
                                   fragment_shader=FRAGMENT_SHADER)
        
        self.view = np.eye(3, dtype=np.float32)
        self.update_projection_matrix()

        self.vbo = None
        self.vao = None

    def set_view(self, view_matrix):
        self.view = view_matrix
        self.update_projection_matrix()

    def update_projection_matrix(self):
        P = ortho_matrix()
        PV = P @ self.view
        
        self.program["u_projection"].write(
            PV.T.astype("f4").tobytes()
        )

    def draw(self, points, color, mode, model=None):
        if model is None:
            model = np.eye(3, dtype=np.float32)
        self.program["u_model"].write(model.T.astype("f4").tobytes())

        if self.vbo is not None:
            self.vbo.release()
        if self.vao is not None:
            self.vao.release()

        self.vbo = self.ctx.buffer(np.asarray(points, dtype="f4").tobytes())
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo,
                                                 "in_position")
        self.program["color"].value = color
        self.vao.render(mode=mode)

    def polygon(self, points, color, model=None):
        pts = np.asarray(points, dtype=np.float32)
        center = np.mean(pts, axis=0)
        vertices = np.vstack([center, pts, pts[0]])
        self.draw(vertices, color, moderngl.TRIANGLE_FAN, model=model)
        self.draw(np.vstack([pts, pts[0]]), (0.05, 0.05, 0.05),
                  moderngl.LINE_STRIP, model=model)

    def line(self, p1, p2, color, model=None):
        points = np.array([p1, p2], dtype=np.float32)
        self.draw(points, color, moderngl.LINES, model=model)

    def circle(self, cx, cy, radius, color, model=None, segments=32):
        t = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
        pts = np.stack(
            [cx + radius * np.cos(t), cy + radius * np.sin(t)], axis=1
        )
        self.polygon(pts, color, model=model)
