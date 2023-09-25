#include "/lib/common.glsl"
#ifdef FSH
in vec2 lrTexCoord;
in mat4 backReprojectionMatrix;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform float near;
uniform float far;
float farPlusNear = far + near;
float farMinusNear = far - near;

uniform sampler2D colortex2;
uniform sampler2D colortex8;
uniform sampler2D colortex10;
uniform sampler2D colortex12;

float GetLinearDepth(float depth) {
	return (2.0 * near) / (farPlusNear - depth * (farMinusNear));
}

void main() {
    vec4 normalDepthData = texelFetch(colortex8, ivec2(gl_FragCoord.xy), 0);
    vec4 prevPos = backReprojectionMatrix * vec4(gl_FragCoord.xy / view * 2 - 1, 1 - 2 * normalDepthData.w, 1);
    prevPos.xyz = 0.5 * prevPos.xyz / prevPos.w + 0.5;
    vec4 prevColor = vec4(0);

    float weight = 1 - 0.1 * length(fract(view * lrTexCoord + 0.5) - 0.5);
    float prevDepth = 1 - texelFetch(colortex2, ivec2(view * prevPos.xy), 0).w;
    if (prevPos.xy == clamp(prevPos.xy, vec2(0), vec2(1))) {
        prevColor = texture(colortex12, prevPos.xy);
        prevColor.a = min(prevColor.a + 0.04, 0.95);
        weight *= prevColor.a;
    } else {
        weight = 0;
    }
    if ((max(abs(prevDepth - prevPos.z),
            abs(GetLinearDepth(prevDepth) - GetLinearDepth(prevPos.z))) > 0.01
            && length(view * prevPos.xy - gl_FragCoord.xy) > 2.5)
         || normalDepthData.a > 1.5
         || length(normalDepthData.rgb) < 0.1) {
        weight = 0;
    }
    /*RENDERTARGETS:12*/
    gl_FragData[0] = vec4(mix(texture(colortex10, lrTexCoord).rgb, prevColor.rgb, weight), prevColor.a / (prevColor.a + 0.04));
}
#endif
#ifdef VSH

out vec2 lrTexCoord;
out mat4 backReprojectionMatrix;
out mat4 forwardReprojectionMatrix;
uniform int frameCounter;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
#define MATERIALMAP_ONLY
#include "/lib/vx/SSBOs.glsl"
void main() {
    backReprojectionMatrix = gbufferPreviousProjection
                           * gbufferPreviousModelView
                           * mat4(vec4(1, 0, 0, 0),
                                  vec4(0, 1, 0, 0),
                                  vec4(0, 0, 1, 0),
                                  vec4(cameraPosition - previousCameraPosition, 1))
                           * gbufferModelViewInverse
                           * gbufferProjectionInverse;
    forwardReprojectionMatrix = gbufferProjection
                              * gbufferModelView
                              * mat4(vec4(1, 0, 0, 0),
                                     vec4(0, 1, 0, 0),
                                     vec4(0, 0, 1, 0),
                                     vec4(previousCameraPosition - cameraPosition, 1))
                              * gbufferPreviousModelViewInverse
                              * gbufferPreviousProjectionInverse;
    
    gl_Position = ftransform();
    lrTexCoord = gl_Position.xy / gl_Position.w * 0.5 + 0.5;
    lrTexCoord = 0.5 * (lrTexCoord - vec2(frameCounter % 2, frameCounter / 2 % 2) / view);
}
#endif