import sys
import glfw
from OpenGL.GL import *

def create_window(width=800, height=600, title="COMP470 Lab 2", vsync=True):
    if not glfw.init():
        sys.exit("GLFW failed to initialise")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    window = glfw.create_window(width, height, title, None, None)
    if not window:
        glfw.terminate()
        sys.exit("GLFW failed to create a window")
    glfw.make_context_current(window)
    glfw.swap_interval(1 if vsync else 0)
    return window

def compile_shader(src, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return shader

def build_program(vertex_src, fragment_src):
    vs = compile_shader(vertex_src, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_src, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

BASIC_VERTEX_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;
void main() {
    gl_Position = vec4(in_position, 1.0);
    gl_PointSize = 6.0;
}
"""

BASIC_FRAGMENT_SRC = """
#version 330 core
uniform vec3 u_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(u_color, 1.0);
}
"""

COLOR_VERTEX_SRC = """
#version 330 core
layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_color;
out vec3 v_color;
void main() {
    gl_Position = vec4(in_position, 1.0);
    v_color = in_color;
}
"""

COLOR_FRAGMENT_SRC = """
#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

def should_close_soon(window, frames_holder, max_frames=180):
    frames_holder[0] += 1
    return glfw.window_should_close(window) or frames_holder[0] >= max_frames