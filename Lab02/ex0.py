import time
import glfw
from OpenGL.GL import *
from common import create_window

VSYNC = True
N_FRAMES = 240

def main():
    window = create_window(title="Task 0 - Sanity Check", vsync=VSYNC)

    # حجز VAO لتوافق كروت Intel Iris Xe مع Core Profile
    dummy_vao = glGenVertexArrays(1)
    glBindVertexArray(dummy_vao)

    vendor = glGetString(GL_VENDOR)
    renderer = glGetString(GL_RENDERER)
    version = glGetString(GL_VERSION)
    
    print(f"Vendor: {vendor.decode() if vendor else 'Unknown'}")
    print(f"Renderer: {renderer.decode() if renderer else 'Unknown'}")
    print(f"GL version: {version.decode() if version else 'Unknown'}")

    glClearColor(0.07, 0.13, 0.23, 1.0)
    frame_times = []
    frame = 0

    while not glfw.window_should_close(window) and frame < N_FRAMES:
        t0 = time.perf_counter()

        glClear(GL_COLOR_BUFFER_BIT)
        glfw.swap_buffers(window)
        glfw.poll_events()

        dt = time.perf_counter() - t0
        frame_times.append(dt)
        frame += 1

    if frame_times:
        avg = sum(frame_times) / len(frame_times)
        print(f"VSYNC={VSYNC} avg frame time={avg*1000:.2f} ms (~{1/avg:.0f} fps)")

    glBindVertexArray(0)
    glDeleteVertexArrays(1, [dummy_vao])
    glfw.terminate()

if __name__ == "__main__":
    main()