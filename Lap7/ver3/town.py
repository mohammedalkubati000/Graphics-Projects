# --- Exercise 2 answers ---
#
# Q1) Which primitive mesh does each element use?
#   ground          -> "quad"    (scale(80, 1, 80))
#   road            -> "quad"    (translate(r, ROAD_LIFT, 0) @ scale(ROAD_WIDTH, 1, 52))
#   wall            -> "cube"    (house() and box() both use the unit cube)
#   roof            -> "roof"
#   pyramid landmark-> "pyramid" (Clock Tower top, Gold Pyramid, Water Tank top)
#
# Q2) box(): M = translate(x, y0+h/2, z) @ rotate_y(yaw) @ scale(w, h, d)
#   Order applied to a unit cube: 1) scale  2) rotate_y  3) translate.
#   scale is applied FIRST, because the vertex is multiplied on the right
#   (M @ p), so the matrix closest to the vertex acts first.
#
# Q3) Why is the roof translated by h and not h/2?
#   The wall is a unit cube centred on the origin (y from -0.5 to +0.5), so it is
#   lifted by h/2 to stand on the ground; its top is then at y = h.
#   The "roof" mesh has its base at y = 0 (LOCAL_BOX: y from 0 to 1), not its
#   centre. So translating by h puts its base exactly on top of the wall.
#   With h/2 the roof would sink halfway into the wall.
#
# Q4) mirror=True makes sx = -w, a negative scale on x, i.e. a reflection.
#   The determinant of the model matrix becomes negative.
#   A negative determinant flips the winding of the triangles: counter-clockwise
#   becomes clockwise after the transform. The GPU then treats front faces as back
#   faces and culls them, so the house would look inside-out.
#   renderer.py computes det(model[:3,:3]) < 0 ("mirrored"). If true (and
#   Renderer.fix_mirrored is on) it sets ctx.front_face = "cw" for that one draw,
#   so the flipped triangles count as front faces again, then it resets
#   front_face = "ccw" right after vao.render so the next object is unaffected.
#
# Q5) LOCAL_BOX stores the bounding box of each mesh in local space
#   (min corner, max corner). In Obj.__init__ it is used to build the 8 corners
#   of that box (all combinations of lo/hi in x, y, z), as homogeneous points
#   (w = 1). They are multiplied by the model matrix once, so self.corners holds
#   the 8 corners of the object's bounding box in WORLD space (an 8x4 array).
#   draw_town does not use it in this lab; it is for a later lab.
#
# Q6) Per frame: set_camera runs ONCE (view + projection uploaded once),
#   draw_town runs ONCE, and inside it renderer.draw runs once per object,
#   each with its own model matrix.
#- Drag: yaw/pitch change, town fixed, camera moves.
#- Scroll: radius changes, target stays (0,1,0).
#- P: ortho = parallel lines stay parallel, no shrinking; persp = shrinking/converging.
#- F/G: fov change = zoom without moving the camera.
#- N at 0.05: z-fighting (flicker); M fixes it. Push near out as far as the scene allows.
#- Pitch unclamped: cross(up, w) = 0 at the pole, look_at can't build u -> NaN/blank screen.

import random
import numpy as np
from mat4 import translate, scale, rotate_y

ROAD_LIFT = 0.01           # roads float this far above the ground
ROAD_WIDTH = 5.0           # wide enough for two cars

# Colors
GROUND = (0.30, 0.36, 0.30)
ROAD = (0.18, 0.19, 0.21)
WALLS = [(0.78, 0.74, 0.66), (0.62, 0.66, 0.72), (0.70, 0.62, 0.55), (0.55, 0.60, 0.58), (0.80, 0.80, 0.78)]
ROOFS = [(0.55, 0.25, 0.20), (0.35, 0.30, 0.42), (0.30, 0.38, 0.45)]


# Stage 4: the local bounding box of each mesh, as (min corner, max corner)
LOCAL_BOX = {
  "cube":    ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
  "roof":    ((-0.5, 0.0, -0.5), (0.5, 1.0, 0.5)),
  "pyramid": ((-0.5, 0.0, -0.5), (0.5, 1.0, 0.5)),
  "quad":    ((-0.5, 0.0, -0.5), (0.5, 0.0, 0.5)),
  "sign":    ((-0.5, -0.5, 0.0), (0.5, 0.5, 0.0)),
  "wheel":   ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)),
}

class Obj:
  def __init__(self, mesh, model, color, name=None):
    self.mesh, self.model, self.color, self.name = mesh, model, color, name
    # the 8 corners of the local box, taken to WORLD space once (nothing moves)
    lo, hi = LOCAL_BOX[mesh]
    local = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) 
                                     for y in (lo[1], hi[1]) 
                                     for z in (lo[2], hi[2])], dtype=np.float32)
    self.corners = (model @ local.T).T             # 8 x 4, w = 1
objects = []

BLOCKS = [-18.0, -6.0, 6.0, 18.0]           # block centres along x and z
ROADS = [-24.0, -12.0, 0.0, 12.0, 24.0]     # road centre-lines

def box(x, z, w, h, d, color, yaw=0.0, y0=0.0, name=None):
    """A unit cube scaled to w x h x d, standing on y = y0, turned by yaw."""
    M = translate(x, y0 + h / 2.0, z) @ rotate_y(yaw) @ scale(w, h, d)
    objects.append(Obj("cube", M, color, name))

def house(x, z, w, h, d, roof_h, wall, roof, yaw=0.0, mirror=False, name=None):
    sx = -w if mirror else w                   # a NEGATIVE scale is a reflection
    base = translate(x, 0, z) @ rotate_y(yaw)
    objects.append(Obj("cube", base @ translate(0, h / 2.0, 0) @ scale(sx, h, d), wall, name))
    objects.append(Obj("roof", base @ translate(0, h, 0) @ scale(sx, roof_h, d), roof, name))

def build_town():
  objects.clear()
  # ground and roads (flat quads)
  objects.append(Obj("quad", scale(80, 1, 80), GROUND))
  for r in ROADS:
    objects.append(Obj("quad", translate(r, ROAD_LIFT, 0) @ scale(ROAD_WIDTH, 1, 52), ROAD))
    objects.append(Obj("quad", translate(0, ROAD_LIFT, r) @ scale(52, 1, ROAD_WIDTH), ROAD))
  rng = random.Random(470)                   # same town every run
  special = {(-6.0, -6.0), (18.0, -18.0), (-18.0, 18.0), (6.0, 6.0)}
  for bx in BLOCKS:
    for bz in BLOCKS:
      if (bx, bz) in special:
        continue
      for _ in range(rng.randint(2, 3)):
        x = bx + rng.uniform(-1.5, 1.5)
        z = bz + rng.uniform(-1.5, 1.5)
        if rng.random() < 0.5:
          box(x, z, rng.uniform(2.5, 3.5), rng.uniform(3.0, 11.0), rng.uniform(2.5, 3.5),
              rng.choice(WALLS), yaw=rng.choice([0, 15, 30, 45]))
        else:
          house(x, z, rng.uniform(3.0, 3.5), rng.uniform(2.0, 3.0), rng.uniform(2.5, 3.0),
                rng.uniform(1.0, 2.2), rng.choice(WALLS), rng.choice(ROOFS),
                yaw=rng.choice([0, 20, 90, 110]))

  # --- the three landmarks (the photo targets in Stage 5) ---------------
  box(-6, -6, 3, 14, 3, (0.72, 0.24, 0.18), name="Clock Tower")
  objects.append(Obj("pyramid", translate(-6, 14, -6) @ scale(3.8, 3.0, 3.8), (0.35, 0.18, 0.15), "Clock Tower"))
  box(-6, -6, 3.2, 1.2, 3.2, (0.95, 0.90, 0.75), y0=10.5, name="Clock Tower")      # the clock band

  objects.append(Obj("pyramid", translate(18, 0, -18) @ rotate_y(45) @ scale(9, 7, 9), (0.86, 0.68, 0.22), "Gold Pyramid"))

  for lx, lz in ((-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)):
    box(-18 + lx, 18 + lz, 0.4, 8, 0.4, (0.40, 0.42, 0.46), name="Water Tank")   # legs
  box(-18, 18, 4.5, 4, 4.5, (0.20, 0.45, 0.80), y0=8, name="Water Tank")
  objects.append(Obj("pyramid", translate(-18, 12, 18) @ scale(4.7, 1.2, 4.7), (0.16, 0.30, 0.55), "Water Tank"))

  # a house near the centre of town
  house(6, 6, 3.4, 3, 3, 2.0, (0.85, 0.82, 0.55), (0.25, 0.45, 0.30), yaw=30)


def draw_town(renderer, visible=None, hidden_color=None):
  
    drawn = 0
    for i, o in enumerate(objects):
        if visible is not None and not visible[i]:
            if hidden_color is None:
                continue
            renderer.draw(o.mesh, o.model, hidden_color)
        else:
            renderer.draw(o.mesh, o.model, o.color)
        drawn += 1
    return drawn