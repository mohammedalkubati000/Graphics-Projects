import math
import pytest
from vec3 import Vec3

def test_add():
    a = Vec3(1, 2, 3)
    b = Vec3(4, 5, 6)
    r = a + b
    assert (r.x, r.y, r.z) == pytest.approx((5, 7, 9))

def test_sub_direction_a_to_b():
    A = Vec3(1, 1, 0)
    B = Vec3(4, 5, 0)
    d = B - A
    assert (d.x, d.y, d.z) == pytest.approx((3, 4, 0))

def test_scalar_multiply():
    v = Vec3(1, -2, 3)
    r = v * 2
    assert (r.x, r.y, r.z) == pytest.approx((2, -4, 6))
    r2 = 2 * v
    assert (r2.x, r2.y, r2.z) == pytest.approx((2, -4, 6))

def test_length_3_4_5_triangle():
    v = Vec3(3, 4, 0)
    assert v.length() == pytest.approx(5.0)

def test_normalized_3_4_5():
    v = Vec3(3, 4, 0)
    u = v.normalized()
    assert (u.x, u.y, u.z) == pytest.approx((0.6, 0.8, 0.0))
    assert u.length() == pytest.approx(1.0)

def test_normalize_zero_vector_raises():
    with pytest.raises(ValueError):
        Vec3(0, 0, 0).normalized()

def test_dot_perpendicular_is_zero():
    x_axis = Vec3(1, 0, 0)
    y_axis = Vec3(0, 1, 0)
    assert x_axis.dot(y_axis) == pytest.approx(0.0)

def test_dot_by_hand_example():
    u = Vec3(2, 3, 6)
    v = Vec3(1, 0, 0)
    assert u.dot(v) == pytest.approx(2.0)

def test_dot_sign_matches_direction():
    same_ish = Vec3(1, 0, 0).dot(Vec3(1, 0.1, 0))
    opposite = Vec3(1, 0, 0).dot(Vec3(-1, 0.1, 0))
    assert same_ish > 0
    assert opposite < 0

def test_angle_to_dot_by_hand_example():
    u = Vec3(2, 3, 6)
    v = Vec3(1, 0, 0)
    theta_deg = u.angle_to(v, degrees=True)
    assert theta_deg == pytest.approx(73.4, abs=0.1)

def test_angle_to_parallel_is_zero():
    a = Vec3(2, 0, 0)
    b = Vec3(5, 0, 0)
    assert a.angle_to(b) == pytest.approx(0.0, abs=1e-9)

def test_angle_to_clamps_floating_point_drift():
    v = Vec3(1.0, 0.0, 0.0)
    u = Vec3(1.0, 1e-16, 0.0).normalized()
    theta = v.angle_to(u)
    assert theta == pytest.approx(0.0, abs=1e-6)

def test_cross_normal_of_triangle_example():
    A, B, C = Vec3(0, 0, 0), Vec3(4, 0, 0), Vec3(0, 3, 0)
    e1 = B - A
    e2 = C - A
    n = e1.cross(e2)
    assert (n.x, n.y, n.z) == pytest.approx((0.0, 0.0, 12.0))
    assert n.length() == pytest.approx(12.0)
    assert n.normalized() == Vec3(0, 0, 1)

def test_cross_anticommutative():
    a = Vec3(1, 0, 0)
    b = Vec3(0, 1, 0)
    ab = a.cross(b)
    ba = b.cross(a)
    assert (ab.x, ab.y, ab.z) == pytest.approx((-ba.x, -ba.y, -ba.z))

def test_cross_parallel_vectors_is_zero():
    a = Vec3(2, 0, 0)
    b = Vec3(5, 0, 0)
    r = a.cross(b)
    assert (r.x, r.y, r.z) == pytest.approx((0, 0, 0), abs=1e-9)

def test_cross_right_hand_rule_basis():
    x = Vec3(1, 0, 0)
    y = Vec3(0, 1, 0)
    z = Vec3(0, 0, 1)
    r = x.cross(y)
    assert (r.x, r.y, r.z) == pytest.approx((z.x, z.y, z.z))