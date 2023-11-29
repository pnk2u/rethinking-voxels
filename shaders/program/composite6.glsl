////////////////////////////////////////
// Complementary Reimagined by EminGT //
////////////////////////////////////////

//Common//
#include "/lib/common.glsl"

//////////Fragment Shader//////////Fragment Shader//////////Fragment Shader//////////
#ifdef FRAGMENT_SHADER

noperspective in vec2 texCoord;

//Uniforms//
uniform float viewWidth, viewHeight;
uniform float far, near;

uniform vec3 cameraPosition, previousCameraPosition;

uniform mat4 gbufferPreviousProjection, gbufferProjectionInverse;
uniform mat4 gbufferPreviousModelView, gbufferModelViewInverse;

uniform mat4 gbufferModelView;

uniform sampler2D colortex12;
uniform sampler2D colortex8;
uniform sampler2D colortex4;
uniform sampler2D colortex2;
uniform sampler2D colortex6;
uniform sampler2D depthtex1;

#ifndef LIGHT_COLORING
uniform sampler2D colortex3;
#endif

//Pipeline Constants//
#include "/lib/pipelineSettings.glsl"

#ifndef LIGHT_COLORING
    const bool colortex3MipmapEnabled = true;
#else
    const bool colortex8MipmapEnabled = true;
#endif

//Common Variables//
vec2 view = vec2(viewWidth, viewHeight);

//Common Functions//
float GetLinearDepth(float depth) {
    return (2.0 * near) / (far + near - depth * (far - near));
}

//Includes//
#ifdef TAA
    #include "/lib/antialiasing/taa.glsl"
#endif

//Program//
void main() {
    #ifndef LIGHT_COLORING
        vec3 color = texelFetch(colortex3, texelCoord, 0).rgb;
    #else
        vec3 color = texelFetch(colortex8, texelCoord, 0).rgb;
    #endif

    vec3 temp = vec3(0.0);
    float depth;

    depth = texelFetch(depthtex1, texelCoord, 0).r;

    #ifdef TAA
        DoTAA(color, temp, depth);
    #endif

    vec4 accumulatedLight = texelFetch(colortex12, texelCoord, 0);
    vec4 normalDepthData = texelFetch(colortex8, texelCoord, 0);
    vec3 newNormalData = texelFetch(colortex4, texelCoord, 0).gba * 2 - 1;
    float guessedDepth = 1 - normalDepthData.a;
    float guessedLinDepth = guessedDepth < 1.0 ? GetLinearDepth(guessedDepth) : 20;
    float actualLinDepth = GetLinearDepth(depth);
    vec3 viewPos = mat3(gbufferProjectionInverse) * vec3(2 * texCoord - 1, depth);
    float vdotn = -dot(normalize(viewPos), mat3(gbufferModelView) * newNormalData);
    if (abs(guessedLinDepth - actualLinDepth) /
        (guessedLinDepth + actualLinDepth) * vdotn > 0.01 ||
        length(newNormalData - normalDepthData.rgb) > 0.1) {
        accumulatedLight.a = 0;
    }
    #ifndef LIGHT_COLORING
    /* RENDERTARGETS:3,2,12 */
    #else
    /* RENDERTARGETS:8,2,12 */
    #endif
	gl_FragData[0] = vec4(color, 1.0);
    gl_FragData[1] = vec4(temp, 1.0 - depth);
    gl_FragData[2] = accumulatedLight;
    
	#ifdef TEMPORAL_FILTER
        #ifndef LIGHT_COLORING
        /* RENDERTARGETS:3,2,12,1 */
        #else
        /* RENDERTARGETS:8,2,12,1 */
        #endif
        gl_FragData[3] = vec4(depth, 0.0, 0.0, 1.0);
	#endif
}

#endif

//////////Vertex Shader//////////Vertex Shader//////////Vertex Shader//////////
#ifdef VERTEX_SHADER

noperspective out vec2 texCoord;

//Uniforms//

//Attributes//

//Common Variables//

//Common Functions//

//Includes//

//Program//
void main() {
    gl_Position = ftransform();

    texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
}

#endif
