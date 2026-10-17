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

  #rng = random.Random(470)                   # same town every run
  #special = {(-6.0, -6.0), (18.0, -18.0), (-18.0, 18.0), (6.0, 6.0)}
  #for bx in BLOCKS:
  #  for bz in BLOCKS:
  #    if (bx, bz) in special:
  #      continue
  #    for _ in range(rng.randint(2, 3)):
  #      x = bx + rng.uniform(-1.5, 1.5)
  #      z = bz + rng.uniform(-1.5, 1.5)
  #      if rng.random() < 0.5:
  #        box(x, z, rng.uniform(2.5, 3.5), rng.uniform(3.0, 11.0), rng.uniform(2.5, 3.5),
  #        rng.choice(WALLS), yaw=rng.choice([0, 15, 30, 45]))
  #      else:
  #        house(x, z, rng.uniform(3.0, 3.5), rng.uniform(2.0, 3.0), rng.uniform(2.5, 3.0),
  #        rng.uniform(1.0, 2.2), rng.choice(WALLS), rng.choice(ROOFS),
  #        yaw=rng.choice([0, 20, 90, 110]))
  # --- the three landmarks (the photo targets in Stage 5) ---------------
  #  box(-6, -6, 3, 14, 3, (0.72, 0.24, 0.18), name="Clock Tower")
  #  objects.append(Obj("pyramid", translate(-6, 14, -6) @ scale(3.8, 3.0, 3.8), (0.35, 0.18, 0.15), "Clock Tower"))
  #  box(-6, -6, 3.2, 1.2, 3.2, (0.95, 0.90, 0.75), y0=10.5, name="Clock Tower")      # the clock band

  #  objects.append(Obj("pyramid", translate(18, 0, -18) @ rotate_y(45) @ scale(9, 7, 9), (0.86, 0.68, 0.22), "Gold Pyramid"))

  #  for lx, lz in ((-1.5, -1.5), (1.5, -1.5), (-1.5, 1.5), (1.5, 1.5)):
  #      box(-18 + lx, 18 + lz, 0.4, 8, 0.4, (0.40, 0.42, 0.46), name="Water Tank")   # legs
  #  box(-18, 18, 4.5, 4, 4.5, (0.20, 0.45, 0.80), y0=8, name="Water Tank")
  #  objects.append(Obj("pyramid", translate(-18, 12, 18) @ scale(4.7, 1.2, 4.7), (0.16, 0.30, 0.55), "Water Tank"))

  # a house near the centre of town
  # house(6, 6, 3.4, 3, 3, 2.0, (0.85, 0.82, 0.55), (0.25, 0.45, 0.30), yaw=30)


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