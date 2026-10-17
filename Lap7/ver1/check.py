import numpy as np
from mat4 import *

np.set_printoptions(precision=4, suppress=True)

# ---------- 1) rotate_z(90) @ (1,0,0) ----------
print("=== 1) rotate_z(90) ===")
Rz = rotate_z(90)
print(Rz)
print("row 2, col 1 (sin) =", round(float(Rz[1, 0]), 4), "(expected +1)")
print("result:", transform_dir(Rz, (1, 0, 0)), "(expected 0 1 0)")

# ---------- 2) rotate_axis((1,1,1), 120) three times ----------
print("\n=== 2) rotate_axis((1,1,1), 120)^3 ===")
R = rotate_axis((1, 1, 1), 120)
print(R)
print("R@R@R:\n", R @ R @ R)
print("allclose to identity:", np.allclose(R @ R @ R, np.eye(4), atol=1e-5))

# ---------- 3) normal matrix ----------
print("\n=== 3) normal_matrix(scale(2,1,1)) ===")
M = scale(2, 1, 1)
n = np.array([1, -1, 0], dtype=np.float32)
tangent = transform_dir(M, (1, 1, 0))       # transformed tangent
N = normal_matrix(M)
n_correct = N @ n
n_wrong = M[:3, :3] @ n                      # upper-left 3x3 directly
print("N =\n", N)
print("transformed tangent:", tangent)
print("normal via N      :", n_correct, " dot =", round(float(np.dot(n_correct, tangent)), 6), "(expected 0)")
print("normal via M[:3,:3]:", n_wrong, " dot =", round(float(np.dot(n_wrong, tangent)), 6), "(NOT 0)")

# ---------- 4) perspective near / far ----------
print("\n=== 4) perspective(50, 1.0, 0.1, 100) ===")
P = perspective(50, 1.0, 0.1, 100)
print(P)
for name, z in (("near", -0.1), ("far", -100.0)):
    clip = to_clip(P, (0, 0, z))
    ndc = clip[:3] / clip[3]
    print(f"{name}: clip = {clip}, w = {clip[3]:.4f}, NDC z = {ndc[2]:.4f}")
print("(expected near = -1, far = +1)")