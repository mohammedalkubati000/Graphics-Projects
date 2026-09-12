import glfw
from OpenGL.GL import *
from common import create_window

bg_color = [0.07, 0.13, 0.23]

def key_callback(window, key, scancode, action, mods):
    global bg_color
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        if bg_color == [0.07, 0.13, 0.23]:
            bg_color = [0.2, 0.5, 0.3]
        else:
            bg_color = [0.07, 0.13, 0.23]

def main():
    window = create_window(title="Task 1 - Keyboard Events")

    glfw.set_key_callback(window, key_callback)

    dummy_vao = glGenVertexArrays(1)
    glBindVertexArray(dummy_vao)

    while not glfw.window_should_close(window):
        glClearColor(bg_color[0], bg_color[1], bg_color[2], 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glBindVertexArray(0)
    glDeleteVertexArrays(1, [dummy_vao])
    glfw.terminate()

if __name__ == "__main__":
    main()