#include "/lib/common.glsl"
#ifdef FSH
in vec2 lrTexCoord;
in mat4 unProjectionMatrix;
in mat4 prevProjectionMatrix;

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
uniform sampler2D colortex10;
uniform sampler2D colortex12;

float GetLinearDepth(float depth) {
	return (2.0 * near) / (farPlusNear - depth * (farMinusNear));
}
#define FALLOFF_SPEED 0.04
void main() {
    vec4 normalDepthData = texelFetch(colortex8, ivec2(gl_FragCoord.xy), 0);
    vec4 playerPos = unProjectionMatrix * vec4(gl_FragCoord.xy / view * 2 - 1, 1 - 2 * normalDepthData.w, 1);
    vec4 prevPlayerPos = vec4(playerPos.xyz / playerPos.w + cameraPosition - previousCameraPosition, 1);
    vec4 prevPos = prevProjectionMatrix * prevPlayerPos;
    float normalWeight = clamp(dot(prevPlayerPos.xyz, normalDepthData.xyz) / dot(playerPos.xyz, normalDepthData.xyz), 0, 1);
    normalWeight *= normalWeight;
    prevPos.xyz = 0.5 * prevPos.xyz / prevPos.w + 0.5;
    vec4 prevColor = vec4(0);

    float weight = clamp(0.9 + 0.2 * length(fract(view * lrTexCoord) - 0.5), 0, 1);
    float prevDepth = 1 - texelFetch(colortex2, ivec2(view * prevPos.xy), 0).w;
    if (prevPos.xy == clamp(prevPos.xy, vec2(0), vec2(1))) {
        prevColor = texture(colortex12, prevPos.xy);
        prevColor.a = clamp(prevColor.a + FALLOFF_SPEED, 0, 0.99 * normalWeight);
    } else {
        prevColor.a = 0;
    }
    if ((max(abs(prevDepth - prevPos.z),
            abs(GetLinearDepth(prevDepth) - GetLinearDepth(prevPos.z))) > 0.0051
            && length(view * prevPos.xy - gl_FragCoord.xy) > 2.5)
         || normalDepthData.a > 1.5
         || length(normalDepthData.rgb) < 0.1) {
        prevColor.a = 0;
    }
    prevColor.a = clamp(1 - 5 * length(cameraPosition - previousCameraPosition), 0.8 * prevColor.a, prevColor.a);
    /*RENDERTARGETS:12*/
    gl_FragData[0] = vec4(
        mix(texture(colortex10, lrTexCoord).rgb,
            prevColor.rgb,
            weight * prevColor.a / (prevColor.a + FALLOFF_SPEED)
            ),
        prevColor.a);
}
#endif
#ifdef VSH

out vec2 lrTexCoord;
out mat4 unProjectionMatrix;
out mat4 prevProjectionMatrix;
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