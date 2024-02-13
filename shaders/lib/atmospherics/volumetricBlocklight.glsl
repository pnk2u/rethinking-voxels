// Volumetric tracing from Robobo1221, highly modified
#include "/lib/vx/irradianceCache.glsl"
#ifndef LIGHTSHAFTS_ACTIVE
    float GetDepth(float depth) {
        return 2.0 * near * far / (far + near - (2.0 * depth - 1.0) * (far - near));
    }

    float GetDistX(float dist) {
        return (far * (dist - near)) / (dist * (far - near));
    }
#endif
vec4 GetVolumetricBlocklight(inout vec3 color, float vlFactor, vec3 translucentMult, float lViewPos, vec3 nViewPos, vec2 texCoord, float z0, float z1, float dither) {
    if (max(blindness, darknessFactor) > 0.1) return vec4(0.0);
    vec4 volumetricLight = vec4(0.0);

    #ifdef OVERWORLD
        float vlSceneIntensity = isEyeInWater != 1 ? vlFactor : 1.0;
        float vlMult = 1.0;


        if (sunVisibility < 0.5) {
            vlSceneIntensity = 0.0;
            vlMult = 0.6 + 0.4 * max0(far - lViewPos) / far;
        }

    #endif
    #if LIGHTSHAFT_QUALI == 4
        int sampleCount = vlSceneIntensity < 0.5 ? 30 : 50;
    #elif LIGHTSHAFT_QUALI == 3
        int sampleCount = vlSceneIntensity < 0.5 ? 15 : 30;
    #elif LIGHTSHAFT_QUALI == 2
        int sampleCount = vlSceneIntensity < 0.5 ? 10 : 20;
    #elif LIGHTSHAFT_QUALI == 1
        int sampleCount = vlSceneIntensity < 0.5 ? 6 : 12;
    #endif
    float addition = 1.0;
    float maxDist = mix(max(far, 96.0) * 0.55, 80.0, vlSceneIntensity);

    #if WATER_FOG_MULT != 100
        if (isEyeInWater == 1) {
            #define WATER_FOG_MULT_M WATER_FOG_MULT * 0.01;
            maxDist /= WATER_FOG_MULT_M;
        }
    #endif

    float distMult = maxDist / (sampleCount + addition);
    float sampleMultIntense = isEyeInWater != 1 ? 1.0 : 1.85;

    float viewFactor = 1.0 - 0.7 * pow2(dot(nViewPos.xy, nViewPos.xy));

    float depth0 = GetDepth(z0);
    float depth1 = GetDepth(z1);

    // Fast but inaccurate perspective distortion approximation
    maxDist *= viewFactor;
    distMult *= viewFactor;

    #ifdef OVERWORLD
        float maxCurrentDist = min(depth1, maxDist);
    #else
        float maxCurrentDist = min(depth1, far);
    #endif

    for (int i = 0; i < sampleCount; i++) {
        float currentDist = (i + dither) * distMult + addition;

        if (currentDist > maxCurrentDist) break;

        vec4 viewPos = gbufferProjectionInverse * (vec4(texCoord, GetDistX(currentDist), 1.0) * 2.0 - 1.0);
        viewPos /= viewPos.w;
        vec4 wpos = gbufferModelViewInverse * viewPos;
        vec3 playerPos = wpos.xyz / wpos.w;
        vec3 vxPos = playerPos + fract(cameraPosition);

        float shadowSample = 1.0;
        vec3 vlSample = vec3(0.0);
        #ifdef REALTIME_SHADOWS

            #ifdef OVERWORLD
                float percentComplete = currentDist / maxDist;
                float sampleMult = mix(percentComplete * 3.0, sampleMultIntense, max(rainFactor, vlSceneIntensity));
                if (currentDist < 5.0) sampleMult *= smoothstep1(clamp(currentDist / 5.0, 0.0, 1.0));
                sampleMult /= sampleCount;
            #endif

            if (infnorm(vxPos/voxelVolumeSize) < 0.5) {
                vlSample = readVolumetricBlocklight(vxPos);
            }
        #endif

        if (currentDist > depth0) vlSample *= translucentMult;

        volumetricLight += vec4(vlSample, shadowSample) * sampleMult;
    }

    #ifdef OVERWORLD
        #if LIGHTSHAFT_DAY_I != 100 || LIGHTSHAFT_NIGHT_I != 100
            #define LIGHTSHAFT_DAY_IM LIGHTSHAFT_DAY_I * 0.01
            #define LIGHTSHAFT_NIGHT_IM LIGHTSHAFT_NIGHT_I * 0.01
            vlMult.rgb *= mix(LIGHTSHAFT_NIGHT_IM, LIGHTSHAFT_DAY_IM, sunVisibility);
        #endif

        #if LIGHTSHAFT_RAIN_I != 100
            #define LIGHTSHAFT_RAIN_IM LIGHTSHAFT_RAIN_I * 0.01
            vlMult.rgb *= mix(1.0, LIGHTSHAFT_RAIN_IM, rainFactor);
        #endif

        volumetricLight.rgb *= vlMult;
    #endif

    volumetricLight = max(volumetricLight, vec4(0.0));
    volumetricLight.a = min(volumetricLight.a, 1.0);

    return volumetricLight;
}