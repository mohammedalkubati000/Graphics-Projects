import math

class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Vec3({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"

    def __eq__(self, other):
        return (
            isinstance(other, Vec3)
            and math.isclose(self.x, other.x, abs_tol=1e-9)
            and math.isclose(self.y, other.y, abs_tol=1e-9)
            and math.isclose(self.z, other.z, abs_tol=1e-9)
        )

    def as_tuple(self):
        return (self.x, self.y, self.z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self):
        len_val = self.length()
        if len_val == 0.0:
            raise ValueError("Cannot normalize a zero-length vector.")
        return Vec3(self.x / len_val, self.y / len_val, self.z / len_val)

    def angle_to(self, other, degrees=False):
        dot_val = self.dot(other)
        mag_product = self.length() * other.length()
        if mag_product == 0.0:
            raise ValueError("Cannot calculate angle with zero-length vector.")
        cos_theta = max(-1.0, min(1.0, dot_val / mag_product))
        rad = math.acos(cos_theta)
        return math.degrees(rad) if degrees else rad
if __name__ == "__main__":
    v1 = Vec3(1, 2, 3)
    v2 = Vec3(4, 5, 6)

    assert v1 + v2 == Vec3(5, 7, 9)
    assert v2 - v1 == Vec3(3, 3, 3)
    assert v1 * 2 == Vec3(2, 4, 6)
    assert 2 * v1 == Vec3(2, 4, 6)
    assert v1.dot(v2) == 32.0
    assert v1.cross(v2) == Vec3(-3, 6, -3)
    assert math.isclose(Vec3(3, 4, 0).length(), 5.0)
    assert Vec3(3, 0, 0).normalized() == Vec3(1, 0, 0)
    assert math.isclose(Vec3(1, 0, 0).angle_to(Vec3(0, 1, 0), degrees=True), 90.0)

    print("All Vec3 assertions passed successfully!")