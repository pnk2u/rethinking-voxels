#include "/lib/common.glsl"

#ifdef CSH
#ifdef PER_PIXEL_LIGHT
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
#if BLOCKLIGHT_RESOLUTION == 1
    const vec2 workGroupsRender = vec2(1.0, 1.0);
#elif BLOCKLIGHT_RESOLUTION == 2
    const vec2 workGroupsRender = vec2(0.5, 0.5);
#elif BLOCKLIGHT_RESOLUTION == 3
    const vec2 workGroupsRender = vec2(0.3333333, 0.3333333);
#elif BLOCKLIGHT_RESOLUTION == 4
    const vec2 workGroupsRender = vec2(0.25, 0.25);
#elif BLOCKLIGHT_RESOLUTION == 6
    const vec2 workGroupsRender = vec2(0.1666667, 0.1666667);
#elif BLOCKLIGHT_RESOLUTION == 8
    const vec2 workGroupsRender = vec2(0.125, 0.125);
#endif
vec2 view = vec2(viewWidth, viewHeight);
layout(rgba16f) uniform image2D colorimg10;
#ifdef BLOCKLIGHT_HIGHLIGHT
    layout(rgba16f) uniform image2D colorimg13;
#endif
layout(rgba16i) uniform iimage2D colorimg11;

vec3 fractCamPos =
    cameraPositionInt.y == -98257195 ?
    fract(cameraPosition) :
    cameraPositionFract;
ivec3 floorCamPosOffset =
    cameraPositionInt.y == -98257195 ?
    ivec3((floor(cameraPosition) - floor(previousCameraPosition)) * 1.001) :
    cameraPositionInt - previousCameraPositionInt;

#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/voxelReading.glsl"
#include "/lib/util/random.glsl"
#include "/lib/vx/positionHashing.glsl"
#ifdef BLOCKLIGHT_HIGHLIGHT
    #include "/lib/lighting/ggx.glsl"
#endif

#if MAX_TRACE_COUNT < 128
    #define MAX_LIGHT_COUNT 128
#else
    #define MAX_LIGHT_COUNT 512
#endif
shared int lightCount;
shared ivec4 cumulatedPos;
shared ivec4 cumulatedNormal;
shared ivec4[MAX_LIGHT_COUNT] positions;
shared int[MAX_LIGHT_COUNT] extraData;
shared float[MAX_LIGHT_COUNT] weights;
shared uint[128] lightHashMap;
shared uvec3 probeLightCols[5];

ivec2 getFlipPair(int index, int stage) {
    int groupSize = 1<<stage;
    return ivec2(index / groupSize * groupSize * 2) +
           ivec2(index%groupSize, 2 * groupSize - index%groupSize - 1);
}
ivec2 getDispersePair(int index, int stage) {
    int groupSize = 1<<stage;
    return ivec2(index / groupSize * groupSize * 2) +
           ivec2(index%groupSize, groupSize + index%groupSize);
}

void flipPair(int index, int stage) {
    ivec2 indexPair = getFlipPair(index, stage);
    if (
        indexPair.y < lightCount && 
        weights[indexPair.x] < weights[indexPair.y]
    ) {
        ivec4 temp = positions[indexPair.x];
        float temp2 = weights[indexPair.x];
        positions[indexPair.x] = positions[indexPair.y];
        positions[indexPair.y] = temp;
        weights[indexPair.x] = weights[indexPair.y];
        weights[indexPair.y] = temp2;
    }
}

void dispersePair(int index, int stage) {
    ivec2 indexPair = getDispersePair(index, stage);
    if (
        indexPair.y < lightCount &&
        weights[indexPair.x] < weights[indexPair.y]
    ) {
        ivec4 temp = positions[indexPair.x];
        float temp2 = weights[indexPair.x];
        positions[indexPair.x] = positions[indexPair.y];
        positions[indexPair.y] = temp;
        weights[indexPair.x] = weights[indexPair.y];
        weights[indexPair.y] = temp2;
    }
}

void main() {
    int index = int(gl_LocalInvocationID.x + gl_WorkGroupSize.x * gl_LocalInvocationID.y);
    float dither = nextFloat();
    if (index == 0) {
        lightCount = 0;
        cumulatedPos = ivec4(0);
        cumulatedNormal = ivec4(0);
    }
    if (index < 128) {
        lightHashMap[index] = 0;
    }
    if (index < 5) {
        probeLightCols[index] = uvec3(0);
    }
    barrier();
    memoryBarrierShared();
    ivec2 readTexelCoord
        = ivec2(gl_GlobalInvocationID.xy) * BLOCKLIGHT_RESOLUTION
        + ivec2(
            BLOCKLIGHT_RESOLUTION
            * fract(vec2(
                frameCounter % 1000 * 1.618033988749895,
                frameCounter % 1000 * 1.618033988749895 * 1.618033988749895
            ) + vec2(
                (gl_GlobalInvocationID.x) * 1.618033988749895 * 1.618033988749895 * 1.618033988749895,
                (gl_GlobalInvocationID.x) * 1.618033988749895 * 1.618033988749895 * 1.618033988749895 * 1.618033988749895
            ) + vec2(
                (gl_GlobalInvocationID.y) * 1.618033988749895 * 1.618033988749895 * 1.618033988749895 * 1.618033988749895 * 1.618033988749895,
                (gl_GlobalInvocationID.y) * 1.618033988749895 * 1.618033988749895 * 1.618033988749895 * 1.618033988749895 * 1.618033988749895 * 1.618033988749895
            ))
        );
    ivec2 writeTexelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec4 normalDepthData = texelFetch(colortex8, readTexelCoord, 0);
    #ifdef BLOCKLIGHT_HIGHLIGHT
        float smoothness = texelFetch(colortex3, readTexelCoord, 0).r;
    #endif
    ivec3 vxPosFrameOffset = -floorCamPosOffset;
    bool validData = (normalDepthData.a < 1.5 && length(normalDepthData.rgb) > 0.1 && all(lessThan(readTexelCoord, ivec2(view + 0.1))));
    for (int k = 1; k < BLOCKLIGHT_RESOLUTION; k++) {
        if (validData) break;
        for (int i = 0; i < 2 * k - 1; i++) {
            ivec2 offset = i < k ? ivec2(i, k) : ivec2(k, 2 * k - 2 - i);
            ivec2 newReadTexelCoord = readTexelCoord + offset;
            normalDepthData = texelFetch(colortex8, newReadTexelCoord, 0);
            bool validData = (normalDepthData.a < 1.5 && length(normalDepthData.rgb) > 0.1 && all(lessThan(newReadTexelCoord, ivec2(view + 0.1))));
            if (validData) {
                readTexelCoord = newReadTexelCoord;
            }
        }
    }
    vec4 playerPos = vec4(1000);
    vec3 vxPos = vec3(1000);
    vec3 biasedVxPos = vec3(1000);
    ivec3 lightStorageCoords = ivec3(-1);
    barrier();
    if (index < MAX_LIGHT_COUNT) {
        ivec4 prevFrameLight = imageLoad(colorimg11, writeTexelCoord);
        prevFrameLight.xyz += vxPosFrameOffset;

        bool isStillLight = (imageLoad(occupancyVolume, prevFrameLight.xyz + voxelVolumeSize/2).r >> 16 & 1) != 0;
        if (!isStillLight && prevFrameLight.w > 0) {
            for (int k = 0; k < 6; k++) {
                ivec3 offset = (k/3*2-1) * ivec3(equal(ivec3(k%3), ivec3(0, 1, 2)));
                if ((imageLoad(occupancyVolume, prevFrameLight.xyz + offset + voxelVolumeSize/2).r >> 16 & 1) != 0) {
                    isStillLight = true;
                    prevFrameLight.xyz += offset;
                    break;
                }
            }
        }
        uint hash = posToHash(prevFrameLight.xyz) % uint(128*32);
        bool known = (
            prevFrameLight.w <= 0 ||
            !isStillLight
        );
        if (!known) {
            known = (atomicOr(lightHashMap[hash/32], uint(1)<<hash%32) & uint(1)<<hash%32) != 0;
        }

        if (!known) {
            int thisLightIndex = atomicAdd(lightCount, 1);
            if (thisLightIndex < MAX_LIGHT_COUNT) {
                positions[thisLightIndex] = ivec4(prevFrameLight.xyz, 0);
                weights[thisLightIndex] = 0.0001 * prevFrameLight.w;
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
    }
    barrier();

    if (validData) {
        playerPos =
            gbufferModelViewInverse *
            (gbufferProjectionInverse *
            (vec4((readTexelCoord + 0.5) / view,
            1 - normalDepthData.a,
            1) * 2 - 1));
        playerPos /= playerPos.w;
        vxPos = playerPos.xyz + fractCamPos;
        #if PIXEL_SHADOW > 0 && !defined GBUFFERS_HAND
            vxPos = floor(vxPos * PIXEL_SHADOW + 0.5 * normalDepthData.xyz) / PIXEL_SHADOW + 0.5 / PIXEL_SHADOW;
        #endif
        float bias = max(0.6/(1<<VOXEL_DETAIL_AMOUNT), 1.2 * infnorm(vxPos/voxelVolumeSize));
        int thisResolution = getVoxelResolution(vxPos);
        if (
            thisResolution != getVoxelResolution(vxPos + 1.0/(1<<thisResolution) * normalDepthData.xyz)
        ) {
            bias = 1.0/(1<<thisResolution);
        }
        float dfValMargin = 0.01;
        if (normalDepthData.a > 0.44) { // hand
            dfValMargin = 0.5;
        }
        for (int k = 0; k < 4; k++) {
            biasedVxPos = vxPos + bias * normalDepthData.xyz;
            vec3 dfGrad = distanceFieldGradient(biasedVxPos);
            if (dfGrad != vec3(0)) dfGrad = normalize(dfGrad);
            vec3 dfGradPerp = dfGrad - dot(normalDepthData.xyz, dfGrad) * normalDepthData.xyz;
            float dfVal = getDistanceField(biasedVxPos);
            float dfGradPerpLength = length(dfGradPerp);
            if (dfGradPerpLength > 0.1) {
                float resolution = min(VOXEL_DETAIL_AMOUNT, -log2(infnorm(abs(vxPos) / voxelVolumeSize) - 0.5));
                dfVal = min(dfVal, getDistanceField(biasedVxPos - dfGradPerp / (pow(2, resolution + 1) * dfGradPerpLength)));
            }
            if (dfVal > dfValMargin) break;
            bias += max(0.01, dfValMargin - dfVal);
        }
        biasedVxPos = vxPos + min(1.1, bias) * normalDepthData.xyz;
        lightStorageCoords = ivec3(biasedVxPos + voxelVolumeSize/2)/8*8;
        ivec4 discretizedVxPos = ivec4(100 * vxPos, 100);
        ivec4 discretizedNormal = ivec4(10 * normalDepthData.xyz, 10);
        for (int i = 0; i < 4; i++) {
            atomicAdd(cumulatedPos[i], discretizedVxPos[i]);
            atomicAdd(cumulatedNormal[i], discretizedNormal[i]);
        }
        vec3 dir = randomSphereSample();
        if (dot(dir, normalDepthData.xyz) < 0) {
            dir *= -1;
        }
        vec3 rayNormal0;
        vec4 rayHit0 = voxelTrace(biasedVxPos, LIGHT_TRACE_LENGTH * dir, rayNormal0);
        ivec3 rayHit0Coords = ivec3(rayHit0.xyz - 0.5 * rayNormal0 + 1000) - 1000;
        int offsetDecider = 26 - int(nextFloat() * 100);
        rayHit0Coords += int(offsetDecider >= 0) * (ivec3(
            offsetDecider%3, offsetDecider/3%3, offsetDecider/9%3
        ) - 1);
        if ((imageLoad(occupancyVolume, rayHit0Coords + voxelVolumeSize/2).r >> 16 & 1) != 0) {
            uint hash = posToHash(rayHit0Coords) % uint(128*32);
            if ((atomicOr(lightHashMap[hash/32], 1<<hash%32) & uint(1)<<hash%32) == 0) {
                int lightIndex = atomicAdd(lightCount, 1);
                if (lightIndex < MAX_LIGHT_COUNT) {
                    positions[lightIndex] = ivec4(rayHit0Coords, 1);
                    vec3 lightPos = rayHit0Coords + 0.5;
                    float ndotl = max(0, dot(normalize(lightPos - vxPos), normalDepthData.xyz));
                    float dirLen = length(lightPos - vxPos);
                    weights[lightIndex] =
                        length(getColor(lightPos).xyz) *
                        ndotl *
                        (sqrt(1 - min(1.0, dirLen / LIGHT_TRACE_LENGTH))) /
                        (dirLen + 0.1);
                } else {
                    atomicMin(lightCount, MAX_LIGHT_COUNT);
                }
            }
        }
    }
    vec3 meanPos = vec3(cumulatedPos.xyz)/cumulatedPos.w;
    vec3 meanNormal = vec3(cumulatedNormal.xyz)/cumulatedNormal.w;
    meanNormal *= length(meanNormal);

    if (index < 125) {
        ivec3 lightPos0 = ivec3(index%5, index/5%5, index/25%5) - 2;
        if ((imageLoad(occupancyVolume, lightPos0 + voxelVolumeSize/2).r >> 16 & 1) != 0) {
            ivec2 packedLightSubPos = ivec2(
                imageLoad(
                    voxelCols,
                    (lightPos0 + voxelVolumeSize/2) * ivec3(1, 2, 1) + ivec3(0, 2 * voxelVolumeSize.y, 0)).r,
                imageLoad(
                    voxelCols,
                    (lightPos0 + voxelVolumeSize/2) * ivec3(1, 2, 1) + ivec3(0, 2 * voxelVolumeSize.y, 0) + ivec3(0, 1, 0)).r
            );
            vec3 subLightPos = 0.1 * vec3(packedLightSubPos.x & 0x7fff, packedLightSubPos.x>>15 & 0x7fff, packedLightSubPos.y & 0x7fff) / (packedLightSubPos.y >>25) - 1;
            uint hash = posToHash(lightPos0) % uint(128*32);
            if ((atomicOr(lightHashMap[hash/32], uint(1)<<hash%32) & uint(1)<<hash%32) == 0) {
                int lightIndex = atomicAdd(lightCount, 1);
                if (lightIndex < MAX_LIGHT_COUNT) {
                    vec3 lightPos = lightPos0 + 0.5;
                    float dirLen = length(lightPos - meanPos);
                    positions[lightIndex] = ivec4(lightPos + 10, 0) - ivec2(10, 0).xxxy;
                    weights[lightIndex] =
                        length(getColor(lightPos).xyz) *
                        (sqrt(1 - min(1.0, dirLen / LIGHT_TRACE_LENGTH))) /
                        (dirLen + 0.1);
                } else {
                    atomicMin(lightCount, MAX_LIGHT_COUNT);
                }
            }
        }
    }

    if (index < 8 * MAX_LIGHT_COUNT) {
        ivec2 offset = (1 + index%8/4*3) * (index%4/2*2-1) * ivec2(index%2, (index+1)%2);
        int otherLightIndex = index / 8;
        ivec4 prevFrameLight = imageLoad(colorimg11, ivec2(gl_WorkGroupSize.xy) * (ivec2(gl_WorkGroupID.xy) + offset) + ivec2(otherLightIndex % gl_WorkGroupSize.x, otherLightIndex / gl_WorkGroupSize.x));
        prevFrameLight.xyz += vxPosFrameOffset;
        uint hash = posToHash(prevFrameLight.xyz) % uint(128*32);
        bool known = (
            prevFrameLight.w <= 0 ||
            (imageLoad(occupancyVolume, prevFrameLight.xyz + voxelVolumeSize/2).r >> 16 & 1) == 0
        );
        if (!known) {
            known = (atomicOr(lightHashMap[hash/32], uint(1)<<hash%32) & uint(1)<<hash%32) != 0;
        }
        if (!known) {
            int thisLightIndex = atomicAdd(lightCount, 1);
            if (thisLightIndex < MAX_LIGHT_COUNT) {
                positions[thisLightIndex] = ivec4(prevFrameLight.xyz, 0);
                weights[thisLightIndex] = 0.000005/(index%8/4+1) * prevFrameLight.w;
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
    }
    barrier();
    memoryBarrierShared();
    bool participateInSorting = index < MAX_LIGHT_COUNT/2;
    #include "/lib/misc/prepare4_BM_sort.glsl"

    if (index < lightCount) {
        extraData[index] = imageLoad(occupancyVolume, positions[index].xyz + voxelVolumeSize/2).r;
    }
    barrier();
    memoryBarrierShared();

    vec3 writeColor = vec3(0);
    #ifdef BLOCKLIGHT_HIGHLIGHT
        vec3 writeSpecular = vec3(0);
    #endif
    uint traceNum = 0;
    if (validData) {
        for (uint thisLightIndex = 0; thisLightIndex < MAX_LIGHT_COUNT; thisLightIndex++) {
            if (thisLightIndex >= lightCount) break;
            uint hash = posToHash(positions[thisLightIndex].xyz) % uint(1<<18);
            uvec2 packedLightSubPos = uvec2(globalLightHashMap[4*hash], globalLightHashMap[4*hash+1]);
            uvec2 packedLightCol = uvec2(globalLightHashMap[4*hash+2], globalLightHashMap[4*hash+3]);
            vec3 subLightPos = 1.0/32.0 * vec3(packedLightSubPos.x & 0xffff, packedLightSubPos.x>>16, packedLightSubPos.y & 0xffff) / (packedLightSubPos.y >> 16) - 1;
            float lightSize = 0.5;
            vec3 lightPos = positions[thisLightIndex].xyz + subLightPos;
            lightSize = clamp(lightSize, 0.01, getDistanceField(lightPos));
            float ndotl0 = max(0.0, dot(normalize(lightPos - vxPos), normalDepthData.xyz));
            vec3 dir = lightPos - biasedVxPos;
            float dirLen = length(dir);
            float thisTraceLen = (extraData[thisLightIndex]>>17 & 31)/32.0;

            if (dirLen < thisTraceLen * LIGHT_TRACE_LENGTH && ndotl0 > 0.001) {
                float lightBrightness = 1.5 * thisTraceLen;
                lightBrightness *= lightBrightness;
                vec4 rayHit1 = coneTrace(biasedVxPos, (1.0 - 0.1 / (dirLen + 0.1)) * dir, lightSize * LIGHTSOURCE_SIZE_MULT / dirLen, dither);
                if (rayHit1.w > 0.01) {
                    vec3 lightColor = 1.0/32.0 * vec3(packedLightCol.x & 0xffff, packedLightCol.x>>16, packedLightCol.y & 0xffff) / (packedLightSubPos.y >> 16);
                    float brightness = (sqrt(1 - dirLen / (LIGHT_TRACE_LENGTH * thisTraceLen))) / (dirLen + 0.1);
                    vec3 thisBaseCol = lightColor * rayHit1.rgb * rayHit1.w * brightness * lightBrightness;
                    writeColor += thisBaseCol * ndotl0;
                    #ifdef BLOCKLIGHT_HIGHLIGHT
                        float specularBrightness = GGX(
                            normalDepthData.xyz,
                            normalize(playerPos.xyz - gbufferModelView[3].xyz),
                            normalize(lightPos - vxPos),
                            ndotl0,
                            smoothness
                        );
                        writeSpecular += thisBaseCol * lightBrightness * specularBrightness;
                    #endif
                    int thisWeight = int(10000.5 * length(thisBaseCol * ndotl0));
                    atomicMax(positions[thisLightIndex].w, thisWeight);
                }
                traceNum++;
                if (traceNum >= MAX_TRACE_COUNT) break;
            }
        }
    }
    barrier();
    memoryBarrierShared();
    float lWriteColor = length(writeColor);
    if (lWriteColor > 0.01) {
        writeColor *= log(lWriteColor+1)/lWriteColor;
    }
    #ifdef BLOCKLIGHT_HIGHLIGHT
        float lWriteSpecular = length(writeSpecular);
        if (lWriteSpecular > 0.01) {
            writeSpecular *= log(lWriteSpecular+1)/lWriteSpecular;
        }
    imageStore(colorimg13, writeTexelCoord, vec4(writeSpecular, 1));
    #endif
    imageStore(colorimg10, writeTexelCoord, vec4(writeColor, 1));
    ivec4 lightPosToStore = (index < lightCount && positions[index].w > 0) ? positions[index] : ivec4(0);
    imageStore(colorimg11, writeTexelCoord, lightPosToStore);
}
#else
    const ivec3 workGroups = ivec3(1, 1, 1);
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {}
#endif
#endif
