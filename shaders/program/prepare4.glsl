#include "/lib/common.glsl"
#ifdef FSH
noperspective in vec2 lrTexCoord;
flat in mat4 unProjectionMatrix;
flat in mat4 prevProjectionMatrix;
#ifdef ACCUMULATION
    uniform float viewWidth;
    uniform float viewHeight;
    vec2 view = vec2(viewWidth, viewHeight);

    uniform float near;
    uniform float far;
    float farPlusNear = far + near;
    float farMinusNear = far - near;

    uniform vec3 cameraPosition;
    uniform vec3 previousCameraPosition;

    uniform sampler2D colortex2;
    uniform sampler2D colortex8;
    const bool colortex10MipmapEnabled = true;
    uniform sampler2D colortex12;
    uniform sampler2D colortex13;
    float GetLinearDepth(float depth) {
        return (2.0 * near) / (farPlusNear - depth * (farMinusNear));
    }
#endif
uniform sampler2D colortex10;
#if DENOISING_DEFINE == 1 && TRACE_ALL_LIGHTS
    #define FALLOFF_SPEED 0.1
#else
    #define FALLOFF_SPEED 0.04
#endif
#define MAX_OLDWEIGHT 0.9
void main() {
    vec4 newColor = texture(colortex10, lrTexCoord);
    #ifdef ACCUMULATION
        vec4 normalDepthData = texelFetch(colortex8, ivec2(gl_FragCoord.xy), 0);
        vec4 playerPos = unProjectionMatrix * vec4(gl_FragCoord.xy / view * 2 - 1, 1 - 2 * normalDepthData.w, 1);
        vec4 prevPlayerPos = vec4(playerPos.xyz / playerPos.w + cameraPosition - previousCameraPosition, 1);
        vec4 prevPos = prevProjectionMatrix * prevPlayerPos;
        float ndotv = dot(normalize(playerPos.xyz), normalDepthData.xyz);
        float normalWeight = clamp(dot(normalize(prevPlayerPos.xyz), normalDepthData.xyz) / ndotv, 1 + ndotv, 1);
        normalWeight *= normalWeight;
        if (normalDepthData.a < 0.44) {
            prevPos.xyz = 0.5 * prevPos.xyz / prevPos.w + 0.5;
            prevPos.xy *= view;
        } else {
            prevPos = vec4(gl_FragCoord.xy, 1 - normalDepthData.a, 1);
        }
        vec4 prevColor = vec4(0);
        float prevLightCount = 0;
        vec4 tex13Data = vec4(0);
        float weight = FALLOFF_SPEED * max(0, 1 - 1.5 * length(fract(view * lrTexCoord) - 0.5));
        float prevDepth = 1 - texture(colortex2, prevPos.xy / view).w;
        float origPrevColora = 0;
        if (prevPos.xy == clamp(prevPos.xy, vec2(0), view)) {
            ivec2 prevCoords = ivec2(prevPos.xy);
            tex13Data = texture(colortex13, prevPos.xy / view);

            prevColor.rgb = texture(colortex12, prevPos.xy / view).rgb;
            prevColor.a = texelFetch(colortex12 , prevCoords, 0).a;
            origPrevColora = prevColor.a;
            prevLightCount = max(floor(prevColor.a + 0.05), 0);
            prevColor.a -= prevLightCount;
            prevColor.a *= normalWeight;
            prevColor.a = clamp(
                MAX_OLDWEIGHT * (
                    1 - 2 * (1 - GetLinearDepth(1 - normalDepthData.a)) * 
                    length(cameraPosition - previousCameraPosition)
                ), 0.8 * prevColor.a, prevColor.a);
        }
        float prevLinDepth = prevDepth < 0.99999 && prevDepth > 0 ? GetLinearDepth(prevDepth) : 20;
        float prevCompareDepth = GetLinearDepth(prevPos.z);
        tex13Data.a = clamp(mix(fract(tex13Data.a), 0.1 * abs(prevLightCount - newColor.a) + 0.05, 0.1), 0.05, 0.95);
        float validMult = float(
            #ifdef RESET_ACCUMULATION_WITHOUT_LIGHTSOURCE
                (((newColor.a > 1.5 || prevLightCount < 0.5 + newColor.a) &&
                    newColor.a > 0.5) ||
                    tex13Data.a > 0.06) &&
            #endif
            (max(abs(prevDepth - prevPos.z),
            abs(prevLinDepth - prevCompareDepth) / (prevLinDepth + prevCompareDepth)) < 0.005// + 0.1 * length(cameraPosition - previousCameraPosition)
            || length(prevPos.xy - gl_FragCoord.xy) < 2.5) &&
            normalDepthData.a < 1.5 &&
            length(normalDepthData.rgb) > 0.1
        );
        prevColor.a *= validMult;
        /*RENDERTARGETS:12,13*/
        gl_FragData[0] = vec4(
            mix(newColor.rgb,
                prevColor.rgb,
                prevColor.a / (prevColor.a + weight + 0.001)
                ),
            min(prevColor.a + weight, MAX_OLDWEIGHT) + floor(newColor.a + 0.1)
        );
        //gl_FragData[0].rgb = vec3(prevColor.a);
        gl_FragData[1] = tex13Data;
    #else
        /*RENDERTARGETS:12*/
        gl_FragData[0] = vec4(newColor.rgb, 0);
    #endif
}
#endif
#ifdef VSH

noperspective out vec2 lrTexCoord;
flat out mat4 unProjectionMatrix;
flat out mat4 prevProjectionMatrix;

uniform int frameCounter;
uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;

void main() {
    unProjectionMatrix = gbufferModelViewInverse
                       * gbufferProjectionInverse;
    prevProjectionMatrix = gbufferPreviousProjection
                         * gbufferPreviousModelView;
    gl_Position = ftransform();
    lrTexCoord = gl_Position.xy / gl_Position.w * 0.5 + 0.5;
    lrTexCoord = 0.5 * (lrTexCoord - vec2(frameCounter % 2, frameCounter / 2 % 2) / view);
}
#endif