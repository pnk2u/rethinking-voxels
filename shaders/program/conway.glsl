#include "/lib/common.glsl"

#ifdef FSH
flat in float falloffFrameMult;

uniform sampler2D colortex14;
uniform float frameTimeCounter;
uniform float viewWidth;
uniform float viewHeight;
ivec2 view = ivec2(viewWidth + 0.5, viewHeight + 0.5);
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
void main() {
    /*RENDERTARGETS:14*/
    float fractTime = fract(3.0 * frameTimeCounter);
    if (ivec2(gl_FragCoord.xy) == ivec2(0)) {
        gl_FragData[0] = vec4(fractTime, 0, 0, 1);
    } else {
        ivec2 texelCoord = ivec2(gl_FragCoord.xy) + ivec2(1.1 * (floor(0.25 * cameraPosition.xz) - floor(0.25 * previousCameraPosition.xz)));
        float prevFractTime = texelFetch(colortex14, ivec2(0), 0).r;
        float writeVal = texelFetch(colortex14, texelCoord, 0).r;
        int liveCount = 0;
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                ivec2 offsetCoord = texelCoord + ivec2(x, y);
                if (offsetCoord != ivec2(0));
                if (texelFetch(colortex14, offsetCoord, 0).r > 0.995) {
                    liveCount++;
                }
            }
        }
        if (liveCount == 3 && prevFractTime > fractTime) {
            writeVal = 1.0;
        } else if ((liveCount != 4 && prevFractTime > fractTime) || writeVal <= 0.995) {
            writeVal *= falloffFrameMult;
        }
        if (all(lessThan(abs(texelCoord - view / 2), ivec2(3)))) {
            writeVal = 1.0;
        }
        gl_FragData[0] = vec4(writeVal, 0, 0, 1);
    }
}
#endif
#ifdef VSH
flat out float falloffFrameMult;

uniform float frameTimeSmooth;

void main() {
    falloffFrameMult = min(0.99, pow(0.05, frameTimeSmooth));
    gl_Position = ftransform();
}
#endif