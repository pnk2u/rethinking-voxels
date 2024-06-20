// this program just flips shadowcolor2
#include "/lib/common.glsl"

#ifdef FRAGMENT_SHADER

void main() {
    #ifdef IRIS_FEATURE_HIGHER_SHADOWCOLOR
        /* RENDERTARGETS:2 */
        gl_FragData[0] = texelFetch(shadowcolor2, ivec2(gl_FragCoord.xy), 0);
    #endif
}
#endif

#ifdef VERTEX_SHADER
void main() {
    #ifdef INTERACTIVE_WATER
        gl_Position = ftransform();
    #else
        gl_Position = vec4(-1);
    #endif
}
#endif