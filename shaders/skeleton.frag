#version 330 core

in vec4 v_color;
out vec4 frag_color;

void main() {
    // For GL_POINTS: circular glow using gl_PointCoord
    // For GL_LINES: simple pass-through (gl_PointCoord is undefined for lines)
    frag_color = v_color;
}
