#version 330 core

in vec4 v_color;
out vec4 frag_color;

uniform int u_ember;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    float glow = 1.0 - smoothstep(0.0, 0.5, dist);
    if (u_ember == 1) {
        // White-hot core fading to particle color at the edge
        float core = 1.0 - smoothstep(0.0, 0.18, dist);
        float mid  = 1.0 - smoothstep(0.1, 0.38, dist);
        vec3 hot   = mix(v_color.rgb, vec3(1.0, 0.92, 0.6), core * 0.85 + mid * 0.25);
        frag_color = vec4(hot, v_color.a * glow);
    } else {
        frag_color = vec4(v_color.rgb, v_color.a * glow);
    }
}
