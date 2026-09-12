import numpy as np
import glfw
from OpenGL.GL import *
from common import create_window, build_program, BASIC_VERTEX_SRC, BASIC_FRAGMENT_SRC

def create_polygon_vertices(n_sides, radius=0.5):
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    vertices = []
    for angle in angles:
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.extend([x, y, 0.0])
    return np.array(vertices, dtype=np.float32)

def main():
    window = create_window(title="Task 2 - Polygon Setup")

    dummy_vao = glGenVertexArrays(1)
    glBindVertexArray(dummy_vao)

    program = build_program(BASIC_VERTEX_SRC, BASIC_FRAGMENT_SRC)
    glUseProgram(program)

    vertices = create_polygon_vertices(n_sides=6, radius=0.6)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    u_color_loc = glGetUniformLocation(program, "u_color")
    glUniform3f(u_color_loc, 0.2, 0.8, 0.4)

    while not glfw.window_should_close(window):
        glClearColor(0.07, 0.13, 0.23, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glDrawArrays(GL_LINE_LOOP, 0, 6)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glBindVertexArray(0)
    glDeleteVertexArrays(1, [dummy_vao])
    glfw.terminate()

if __name__ == "__main__":
    main()