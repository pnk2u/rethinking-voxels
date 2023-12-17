#include "/lib/common.glsl"
#ifdef FSH
#ifdef DENOISING
    uniform int frameCounter;

    uniform float viewWidth;
    uniform float viewHeight;
    vec2 view = vec2(viewWidth, viewHeight);

    uniform float near;
    uniform float far;
    float farPlusNear = far + near;
    float farMinusNear = far - near;

    uniform sampler2D colortex8;
    uniform vec3 cameraPosition;
    #define DENOISE_DATA
    #include "/lib/vx/SSBOs.glsl"
#endif
uniform sampler2D colortex13;
#ifdef FIRST
    uniform sampler2D colortex12;
    #ifdef DENOISING
        #define LIGHT_SAMPLER colortex12
    #endif
#elif defined DENOISING
    #define LIGHT_SAMPLER colortex13
#endif
#ifdef DENOISING
float GetLinearDepth(float depth) {
    return (2.0 * near) / (farPlusNear - depth * (farMinusNear));
}

ivec2[2] readBounds = ivec2[2](ivec2(0), ivec2(view));

void main() {
    ivec2 texelCoord = ivec2(gl_FragCoord.xy);
    vec4 prevTex13Data = texelFetch(colortex13, texelCoord, 0);

    vec4 normalDepthData = texelFetch(colortex8, texelCoord, 0);
    normalDepthData.w = 50.0 * GetLinearDepth(1 - normalDepthData.w);
    vec4 thisLightData = texelFetch(LIGHT_SAMPLER, texelCoord, 0);
    #ifdef FIRST
        float accumulationAmount = 2 * thisLightData.a / ACCUM_FALLOFF_SPEED;
        float variance = denoiseSecondMoment[
            texelCoord.x + 
            int(viewWidth + 0.5) * (
                texelCoord.y + int(viewHeight + 0.5) * (frameCounter % 2)
            )
        ] - pow2(dot(thisLightData.rgb, vec3(1)));
        variance /= max(0.01, 2 * accumulationAmount);
        #if DENOISE_LENIENCE > 0
            float spatialFactor = 1 - (accumulationAmount - 1);
            if (spatialFactor > 0) {
            #endif
            float spatialMean = 0;
            float spatialMoment = 0;
            vec3 maxAroundCol = vec3(0);
            {
                float totalWeight = 0;
                for (int k = 0; k < 25; k++) {
                    ivec2 offset = 3 * (ivec2(k%5, k/5) - 2);
                    float weight = 1.0 / (1 + length(offset));
                    vec3 thisCol = texelFetch(LIGHT_SAMPLER, texelCoord + offset, 0).rgb;
                    if (offset != ivec2(0) && length(offset) < 5) {
                        maxAroundCol = max(maxAroundCol, thisCol);
                    }
                    float thisVal = dot(thisCol, vec3(1));
                    spatialMean += thisVal * weight;
                    spatialMoment += thisVal * thisVal * weight;
                    totalWeight += weight;
                }
                thisLightData.xyz = min(maxAroundCol, thisLightData.xyz);
                spatialMoment /= totalWeight;
                spatialMean /= totalWeight;
                variance = 2 * (spatialMoment - spatialMean * spatialMean);
            }
            #if DENOISE_LENIENCE > 0
            }
        #endif

    #else
        float variance = thisLightData.a;
    #endif
    variance = max(variance, 0.00001);
    float totalWeight = 0.00001;
    vec3 totalLight = vec3(0.0);
    int blurSize1 = int(sqrt(DENOISE_MAX_BLUR_MOD));
    #ifdef FIRST
        blurSize1 = DENOISE_MAX_BLUR_MOD / blurSize1;
    #endif
    blurSize1 *= 2;
    for (int k = 0; k < blurSize1 * blurSize1; k++) {

        ivec2 offset = 2 * ivec2(k%blurSize1, k/blurSize1) - blurSize1;
        #ifndef FIRST
            offset *= blurSize1/2;
        #endif
        if (any(lessThan(offset, -texelCoord + readBounds[0])) || any(greaterThan(offset, readBounds[1] - texelCoord))) {
            continue;
        };
        vec4 aroundNormalDepthData = texelFetch(colortex8, texelCoord + offset, 0);
        aroundNormalDepthData.w = 50.0 * GetLinearDepth(1 - aroundNormalDepthData.w);
        vec4 aroundLight = texelFetch(LIGHT_SAMPLER, texelCoord + offset, 0);
        float weight = exp(-dot(offset, offset) * (2.0 / (DENOISE_MAX_BLUR_MOD * DENOISE_MAX_BLUR_MOD)) - pow2(dot(aroundLight.rgb - thisLightData.rgb, vec3(1))) / (max(DENOISE_LENIENCE * DENOISE_LENIENCE, 1) * variance)) * max0(1 - 7 * length(normalDepthData - aroundNormalDepthData));
        totalWeight += weight;
        totalLight += aroundLight.xyz * weight;
    }
    /*RENDERTARGETS:13*/
    gl_FragData[0] = vec4(totalLight / totalWeight, variance);
//  #ifdef FIRST
//  gl_FragData[0] = vec4(accumulationAmount * 0.01, 0, 0, 1);
//  #endif
}
#elif defined FIRST
void main() {
    ivec2 texelCoord = ivec2(gl_FragCoord.xy);
    vec4 prevTex13Data = texelFetch(colortex13, texelCoord, 0);
    /*RENDERTARGETS:13*/
    gl_FragData[0] = vec4(texelFetch(colortex12, texelCoord, 0).rgb, prevTex13Data.a);
}
#else
    void main() {
        discard;
    }
#endif
#endif
#ifdef VSH
void main() {
    gl_Position = ftransform();
}
#endif
