import glfw
import moderngl
import town
from camera import OrbitCamera
from renderer import Renderer

WINDOW_WIDTH, WINDOW_HEIGHT = 1100, 800
fb_width, fb_height = WINDOW_WIDTH, WINDOW_HEIGHT

camera = OrbitCamera(target=(0.0, 1.0, 0.0), yaw=35.0, pitch=28.0, radius=55.0)
dragging = False
last_mouse = (0.0, 0.0)


def on_framebuffer_resize(window, width, height):
    global fb_width, fb_height
    fb_width, fb_height = max(width, 1), max(height, 1)


def on_key(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return
    if key == glfw.KEY_R:
        camera.reset()
    elif key == glfw.KEY_P:
        camera.orthographic = not camera.orthographic


def on_mouse_button(window, button, action, mods):
    global dragging, last_mouse
    if button == glfw.MOUSE_BUTTON_LEFT:
        dragging = action == glfw.PRESS
        last_mouse = glfw.get_cursor_pos(window)


def on_cursor(window, x, y):
    global last_mouse
    if dragging:
        camera.orbit(-0.3 * (x - last_mouse[0]), 0.3 * (y - last_mouse[1]))
    last_mouse = (x, y)


def on_scroll(window, dx, dy):
    camera.zoom(0.9 if dy > 0 else 1.1)


def process_input(window):
    key = lambda k: glfw.get_key(window, k) == glfw.PRESS
    if key(glfw.KEY_ESCAPE):
        glfw.set_window_should_close(window, True)

    # fov / near / far tuning - one number each, matrices rebuilt next frame
    if key(glfw.KEY_F):
        camera.fov = max(10.0, camera.fov - 0.5)
    if key(glfw.KEY_G):
        camera.fov = min(120.0, camera.fov + 0.5)
    if key(glfw.KEY_N):
        camera.near = max(0.05, camera.near - 0.05)
    if key(glfw.KEY_M):
        camera.near = min(camera.far - 0.1, camera.near + 0.05)
    if key(glfw.KEY_B):
        camera.far = max(camera.near + 0.1, camera.far - 0.5)
    if key(glfw.KEY_V):
        camera.far = min(500.0, camera.far + 0.5)


# -------------------------------------------------------------------- drawing
def draw_frame(ctx, renderer, width, height):
    ctx.viewport = (0, 0, width, height)
    ctx.clear(0.55, 0.70, 0.85, 1.0, depth=1.0)
    ctx.enable(moderngl.CULL_FACE)

    # THE WHOLE STAGE-1 STORY, IN TWO LINES:
    V = camera.view()                                    # rebuild from state
    P = camera.projection(width / height)                # rebuild from state
    renderer.set_camera(V, P)

    town.draw_town(renderer)


# ---------------------------------------------------------------- window setup
def initialize_window():
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.DEPTH_BITS, 24)
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT,
                                "City Driver - Stage 1: Orbit Camera", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    glfw.set_framebuffer_size_callback(window, on_framebuffer_resize)
    glfw.set_key_callback(window, on_key)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor)
    glfw.set_scroll_callback(window, on_scroll)
    return window


def main():
    global fb_width, fb_height
    window = initialize_window()
    fb_width, fb_height = glfw.get_framebuffer_size(window)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    renderer = Renderer(ctx)
    town.build_town()

    while not glfw.window_should_close(window):
        process_input(window)
        draw_frame(ctx, renderer, fb_width, fb_height)
        e = camera.eye()
        glfw.set_window_title(
            window,
            f"Stage 1 | {camera.describe()} | "
            f"eye=({e[0]:6.1f},{e[1]:6.1f},{e[2]:6.1f}) "
            f"yaw={camera.yaw:6.1f} pitch={camera.pitch:6.1f}"
        )
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()