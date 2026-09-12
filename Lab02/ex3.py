import ctypes
import numpy as np
import glfw
from OpenGL.GL import *
from common import create_window, build_program, COLOR_VERTEX_SRC, COLOR_FRAGMENT_SRC

def main():
    window = create_window(title="Task 3 - Interleaved Attributes")
    program = build_program(COLOR_VERTEX_SRC, COLOR_FRAGMENT_SRC)

    vertices = np.array([
        0.0,  0.6, 0.0,  1.0, 0.0, 0.0,
       -0.6, -0.6, 0.0,  0.0, 1.0, 0.0,
        0.6, -0.6, 0.0,  0.0, 0.0, 1.0,
    ], dtype=np.float32)

    stride = 6 * 4

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    glBindVertexArray(0)

    glClearColor(0.07, 0.13, 0.23, 1.0)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(program)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glBindVertexArray(0)
    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glfw.terminate()

if __name__ == "__main__":
    main()