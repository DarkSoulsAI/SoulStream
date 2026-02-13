#version 330 core

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec3 in_color;
layout(location = 2) in float in_alpha;
layout(location = 3) in float in_size;

out vec4 v_color;

void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    gl_PointSize = in_size;
    v_color = vec4(in_color, in_alpha);
}
