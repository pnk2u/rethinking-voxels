vec3 GetColoredLightFog(vec3 nPlayerPos, vec3 translucentMult, float lViewPos, float lViewPos1, float dither, float caveFactor) {
    caveFactor = 1.0;
    vec3 lightFog = vec3(0.0);

    float stepMult = 8.0;

    float maxDist = min(voxelVolumeSize.x * 0.5, far);
    float halfMaxDist = maxDist * 0.5;
    int sampleCount = int(maxDist / stepMult + 0.001);
    vec3 traceAdd = nPlayerPos * stepMult;
    vec3 tracePos = traceAdd * dither;

    vec3 fractCamPos = cameraPositionInt.y == -98257195 ? fract(cameraPosition) : cameraPositionFract;

    for (int i = 0; i < sampleCount; i++) {
        tracePos += traceAdd;

        float lTracePos = length(tracePos);
        if (lTracePos > lViewPos1) break;

        vec3 voxelPos = tracePos + fractCamPos;

        vec3 lightSample = readVolumetricBlocklight(voxelPos);

        float lTracePosM = length(vec3(tracePos.x, tracePos.y * 2.0, tracePos.z));
        lightSample *= max0(1.0 - lTracePosM / maxDist);
        lightSample *= pow2(min1(lTracePos * 0.03125));
        lightSample *= 40 * log(1 + 5 * lightSample * dot(lightSample, lightSample));

        #ifdef CAVE_SMOKE
            if (caveFactor > 0.00001) {
                vec3 smokePos = 0.0025 * (tracePos + cameraPosition);
                vec3 smokeWind = frameTimeCounter * vec3(0.006, 0.003, 0.0);
                float smoke = Noise3D(smokePos + smokeWind)
                            * Noise3D(smokePos * 3.0 - smokeWind)
                            * Noise3D(smokePos * 9.0 + smokeWind);
                smoke = smoothstep1(smoke);
                lightSample *= mix(1.0, smoke * 16.0, caveFactor);
                lightSample += caveFogColor * pow2(smoke) * 0.05 * caveFactor;
            }
        #endif

        if (lTracePos > lViewPos) lightSample *= translucentMult;
        lightFog += lightSample;
    }

    #ifdef NETHER
        lightFog *= netherColor * 2 * VBL_NETHER_MULT;
    #elif defined END
        lightFog *= VBL_END_MULT;
    #endif

    lightFog *= 1.0 - maxBlindnessDarkness;

    return pow(lightFog / sampleCount, vec3(0.25));
}