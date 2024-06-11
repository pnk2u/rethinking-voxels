//////Fragment Shader//////Fragment Shader//////
#ifdef FRAGMENT_SHADER

layout(r32ui) uniform writeonly uimage2D colorimg9;

void main() {
    imageStore(colorimg9, ivec2(gl_FragCoord.xy), uvec4(1<<31));
    /*RENDERTARGETS:8*/
    gl_FragData[0] = vec4(2);
}
#endif

//////Vertex Shader//////Vertex Shader//////
#ifdef VERTEX_SHADER
void main() {
    gl_Position = ftransform();
}
#endif
