#version 330 core

in vec4 v_color;
out vec4 frag_color;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    float glow = 1.0 - smoothstep(0.0, 0.5, dist);
    frag_color = vec4(v_color.rgb, v_color.a * glow);
}
