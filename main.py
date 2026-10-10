import glfw
import moderngl
import numpy as np
import robot
from renderer import Renderer
from camera import Camera2D

is_grabbed = False

def check_grab():
    _, _, _, Mhand = robot.get_local_matrices()
    try:
        Mhand_inv = np.linalg.inv(Mhand)
    except np.linalg.LinAlgError:
        return False

    p_world = np.array([robot.ball_x, robot.ball_y, 1.0])
    p_local = Mhand_inv @ p_world

    gap = robot.gripper_opening * robot.MAX_GRIPPER_GAP
    in_length = (0.4 <= p_local[0] <= 1.0)
    in_width  = (abs(p_local[1]) <= (gap / 2.0 + robot.BALL_RADIUS))

    return in_length and in_width

def process_input(window, camera):
    global is_grabbed

    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        robot.theta1 += 1.0
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        robot.theta1 -= 1.0

    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        robot.theta2 += 1.0
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        robot.theta2 -= 1.0

    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        robot.theta3 += 1.0
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        robot.theta3 -= 1.0

    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        robot.gripper_opening = min(1.0, robot.gripper_opening + 0.02)
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        robot.gripper_opening = max(0.0, robot.gripper_opening - 0.02)

    if glfw.get_key(window, glfw.KEY_J) == glfw.PRESS:
        camera.position[0] -= 0.05
    if glfw.get_key(window, glfw.KEY_L) == glfw.PRESS:
        camera.position[0] += 0.05
    if glfw.get_key(window, glfw.KEY_I) == glfw.PRESS:
        camera.position[1] += 0.05
    if glfw.get_key(window, glfw.KEY_K) == glfw.PRESS:
        camera.position[1] -= 0.05

    if glfw.get_key(window, glfw.KEY_EQUAL) == glfw.PRESS:
        camera.zoom *= 1.02
    if glfw.get_key(window, glfw.KEY_MINUS) == glfw.PRESS:
        camera.zoom /= 1.02

    if glfw.get_key(window, glfw.KEY_COMMA) == glfw.PRESS:
        camera.rotation += 1.0
    if glfw.get_key(window, glfw.KEY_PERIOD) == glfw.PRESS:
        camera.rotation -= 1.0

    if robot.gripper_opening < 0.6 and check_grab():
        is_grabbed = True
    elif robot.gripper_opening >= 0.6:
        is_grabbed = False

    if is_grabbed:
        _, _, _, Mhand = robot.get_local_matrices()
        p_hand_center = Mhand @ np.array([0.72, 0.0, 1.0])
        robot.ball_x = p_hand_center[0]
        robot.ball_y = p_hand_center[1]

def main():
    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 800, "COMP470 - Lab 06 Camera & Robot", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    ctx = moderngl.create_context()

    camera = Camera2D()
    renderer = Renderer(ctx)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        process_input(window, camera)
        renderer.set_view(camera.view_matrix())

        ctx.clear(0.08, 0.09, 0.11, 1.0)

        robot.draw_robot(renderer)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    main()
