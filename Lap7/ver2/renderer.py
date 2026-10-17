"""
renderer.py  -  a small 3D renderer for Lab 7 (provided, read it but you do
not need to change it).

Compared with the 2D renderer from the robot-arm lab:
  * positions are vec3, and every mesh also carries a NORMAL per vertex;
  * the vertex shader receives three mat4 uniforms (model, view, projection)
    and one mat3 (normal_matrix);
  * the depth test is on, so draw order no longer matters.
"""
import numpy as np
import moderngl
from mat4 import gl_bytes, normal_matrix, identity

VERTEX_SHADER = """
#version 330
in vec3 in_position;
in vec3 in_normal;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normal_matrix;
out vec3 v_normal;
void main()
{
    v_normal = normal_matrix * in_normal;          // world-space normal
    gl_Position = projection * view * model * vec4(in_position, 1.0);
    // NOTE: no divide by w here - the GPU clips first, then divides.
}
"""

FRAGMENT_SHADER = """
#version 330
in vec3 v_normal;
uniform vec3 color;
uniform int  lit;
uniform vec3 light_dir;                            // direction TO the light
out vec4 fragColor;
void main()
{
    if (lit == 0) { fragColor = vec4(color, 1.0); return; }
    vec3 n = normalize(v_normal);
    float diffuse = max(dot(n, normalize(light_dir)), 0.0);
    float sky = 0.5 + 0.5 * n.y;                   // a little light from above
    vec3 c = color * (0.22 + 0.13 * sky + 0.70 * diffuse);
    fragColor = vec4(c, 1.0);
}
"""


# ------------------------------------------------------------------ meshes
def _face(verts, normal):
    """Triangulate a convex face and make its winding CCW seen from outside."""
    verts = [np.asarray(v, dtype=np.float32) for v in verts]
    n = np.asarray(normal, dtype=np.float32)
    n = n / np.linalg.norm(n)
    if np.dot(np.cross(verts[1] - verts[0], verts[2] - verts[0]), n) < 0:
        verts = verts[::-1]
    out = []
    for i in range(1, len(verts) - 1):
        for v in (verts[0], verts[i], verts[i + 1]):
            out.extend([*v, *n])
    return out


def cube_data():
    """Unit cube centred on the origin, side 1."""
    h = 0.5
    d = []
    for axis in range(3):
        for sign in (-1, 1):
            n = [0, 0, 0]
            n[axis] = sign
            a, b = [i for i in range(3) if i != axis]
            quad = []
            for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0, 0, 0]
                p[axis], p[a], p[b] = sign * h, s * h, t * h
                quad.append(p)
            d += _face(quad, n)
    return np.array(d, dtype=np.float32)


def quad_data():
    """Unit square in the xz-plane at y = 0, facing up (+y)."""
    h = 0.5
    return np.array(_face([(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)], (0, 1, 0)), dtype=np.float32)


def sign_data():
    """Unit square in the xy-plane, facing +z, ONE-SIDED (for the billboard)."""
    h = 0.5
    return np.array(_face([(-h, -h, 0), (h, -h, 0), (h, h, 0), (-h, h, 0)], (0, 0, 1)), dtype=np.float32)


def roof_data():
    """Gable roof: base [-0.5,0.5] x [-0.5,0.5] at y = 0, ridge along z at y = 1.
    Its two slopes have SLANTED normals - which is what makes the inverse
    transpose visible when the roof is scaled non-uniformly."""
    h = 0.5
    d = []
    d += _face([(-h, 0, -h), (-h, 0, h), (0, 1, h), (0, 1, -h)], (-1, 0.5, 0))    # left slope
    d += _face([(h, 0, -h), (h, 0, h), (0, 1, h), (0, 1, -h)], (1, 0.5, 0))       # right slope
    d += _face([(-h, 0, h), (h, 0, h), (0, 1, h)], (0, 0, 1))                     # front gable
    d += _face([(-h, 0, -h), (h, 0, -h), (0, 1, -h)], (0, 0, -1))                 # back gable
    return np.array(d, dtype=np.float32)


def pyramid_data():
    """Square pyramid: base [-0.5,0.5]^2 at y = 0, apex at (0, 1, 0)."""
    h = 0.5
    apex = (0, 1, 0)
    d = []
    for (x0, z0), (x1, z1) in (((-h, h), (h, h)), ((h, h), (h, -h)), ((h, -h), (-h, -h)), ((-h, -h), (-h, h))):
        mx, mz = (x0 + x1) / 2, (z0 + z1) / 2
        d += _face([(x0, 0, z0), (x1, 0, z1), apex], (mx, 0.25, mz))   # perpendicular to the slope
    d += _face([(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)], (0, -1, 0))
    return np.array(d, dtype=np.float32)


def wheel_data(sides=14):
    """Cylinder along z (the axle), radius 0.5, length 1, FLAT-shaded sides
    so you can see it turn."""
    d = []
    for i in range(sides):
        a0, a1 = 2 * np.pi * i / sides, 2 * np.pi * (i + 1) / sides
        p0 = (0.5 * np.cos(a0), 0.5 * np.sin(a0))
        p1 = (0.5 * np.cos(a1), 0.5 * np.sin(a1))
        am = (a0 + a1) / 2
        d += _face([(p0[0], p0[1], -0.5), (p1[0], p1[1], -0.5), (p1[0], p1[1], 0.5), (p0[0], p0[1], 0.5)],
                   (np.cos(am), np.sin(am), 0))
        d += _face([(0, 0, 0.5), (p0[0], p0[1], 0.5), (p1[0], p1[1], 0.5)], (0, 0, 1))
        d += _face([(0, 0, -0.5), (p1[0], p1[1], -0.5), (p0[0], p0[1], -0.5)], (0, 0, -1))
    return np.array(d, dtype=np.float32)


# ---------------------------------------------------------------- renderer
class Renderer:
    use_normal_matrix = True
    fix_mirrored = True

    def __init__(self, ctx):
        self.ctx = ctx
        self.program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.meshes = {}
        for name, data in (("cube", cube_data()), ("quad", quad_data()), ("sign", sign_data()),
                           ("roof", roof_data()), ("pyramid", pyramid_data()), ("wheel", wheel_data())):
            vbo = ctx.buffer(data.tobytes())
            vao = ctx.vertex_array(self.program, [(vbo, "3f 3f", "in_position", "in_normal")])
            self.meshes[name] = (vao, len(data) // 6)
        # dynamic buffer for lines (frustums, axes, markers)
        self.line_vbo = ctx.buffer(reserve=64 * 1024)
        self.line_vao = ctx.vertex_array(self.program, [(self.line_vbo, "3f", "in_position")])
        self.program["light_dir"].value = tuple(np.array([0.45, 0.80, 0.35]) / np.linalg.norm([0.45, 0.80, 0.35]))
        self.set_camera(identity(), identity())

    def set_camera(self, view, projection):
        """Upload V and P once per viewport, not once per object."""
        self.program["view"].write(gl_bytes(view))
        self.program["projection"].write(gl_bytes(projection))

    def draw(self, mesh, model, color):
        vao, count = self.meshes[mesh]
        model = np.asarray(model, dtype=np.float32)
        self.program["model"].write(gl_bytes(model))
        if Renderer.use_normal_matrix:
            N = normal_matrix(model)                 # inverse transpose
        else:
            N = model[:3, :3]                        # the tempting, wrong choice
        self.program["normal_matrix"].write(gl_bytes(N))
        self.program["color"].value = tuple(color)
        self.program["lit"].value = 1
        # A mirror (det < 0) turns every CCW triangle into CW on screen.
        mirrored = np.linalg.det(model[:3, :3]) < 0
        if mirrored and Renderer.fix_mirrored:
            self.ctx.front_face = "cw"
        vao.render(moderngl.TRIANGLES, vertices=count)
        self.ctx.front_face = "ccw"

    def lines(self, points, color, width=1.0):
        """points: N x 3 array, consecutive pairs are segments (world space)."""
        data = np.asarray(points, dtype="f4").tobytes()
        self.line_vbo.orphan(64 * 1024)
        self.line_vbo.write(data)
        self.program["model"].write(gl_bytes(identity()))
        self.program["color"].value = tuple(color)
        self.program["lit"].value = 0
        self.line_vao.render(moderngl.LINES, vertices=len(points))