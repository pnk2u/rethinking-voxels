#include "/lib/common.glsl"

#ifdef CSH
#ifdef PER_PIXEL_LIGHT
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
const vec2 workGroupsRender = vec2(0.5, 0.5);

uniform int frameCounter;
uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform sampler2D colortex8;
layout(rgba16f) uniform image2D colorimg10;
layout(rgba16i) uniform iimage2D colorimg11;

#include "/lib/vx/voxelReading.glsl"
#include "/lib/util/random.glsl"

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

uint posToHash(uvec3 pos) {
    // modified version of David Hoskins' hash without sine 2
    // https://www.shadertoy.com/view/XdGfRR -> common -> hash13
    // licensed as CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
    pos *= uvec3(1597334673U, 3812015801U, 2798796415U);
    uint hash = (pos.x ^ pos.y ^ pos.z) * 1597334673U;
    return hash % uint(128*32);
}

uint posToHash(vec3 pos) {
    return posToHash(uvec3(pos + 1000));
}

uint posToHash(ivec3 pos) {
    return posToHash(uvec3(pos + 1000));
}

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
    barrier();
    memoryBarrierShared();
    ivec2 readTexelCoord = ivec2(gl_GlobalInvocationID.xy) * 2;// + ivec2(frameCounter % 2, frameCounter / 2 % 2);
    ivec2 writeTexelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec4 normalDepthData = texelFetch(colortex8, readTexelCoord, 0);
    ivec3 vxPosFrameOffset = ivec3((floor(previousCameraPosition) - floor(cameraPosition)) * 1.1);
    bool validData = (normalDepthData.a < 1.5 && length(normalDepthData.rgb) > 0.1 && all(lessThan(readTexelCoord, ivec2(view + 0.1))));
    vec3 vxPos = vec3(1000);
    vec3 biasedVxPos = vec3(1000);
    ivec3 lightStorageCoords = ivec3(-1);
    barrier();
    if (index < MAX_LIGHT_COUNT) {
        ivec4 prevFrameLight = imageLoad(colorimg11, writeTexelCoord);
        prevFrameLight.xyz += vxPosFrameOffset;

        uint hash = posToHash(prevFrameLight.xyz);
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
                weights[thisLightIndex] = 0.0001 * prevFrameLight.w;
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
    }
    barrier();

    if (validData) {
        vec4 playerPos = gbufferModelViewInverse * (gbufferProjectionInverse * (vec4((readTexelCoord + 0.5) / view, 1 - normalDepthData.a, 1) * 2 - 1));
        playerPos /= playerPos.w;
        vxPos = playerPos.xyz + fract(cameraPosition);
        #if PIXEL_SHADOW > 0 && !defined GBUFFERS_HAND
            vxPos = floor(vxPos * PIXEL_SHADOW + 0.5 * normalDepthData.xyz) / PIXEL_SHADOW + 0.5 / PIXEL_SHADOW;
        #endif
        biasedVxPos = vxPos + max(0.03, 1.2 * infnorm(vxPos/voxelVolumeSize)) * normalDepthData.xyz;
        for (int k = 0; k < 4; k++) {
            float dfVal = getDistanceField(biasedVxPos);
            if (dfVal > 0.01) break;
            biasedVxPos += max(0.01, -dfVal) * normalDepthData.xyz;
        }
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
        if (rayHit0.a > 16 && (imageLoad(occupancyVolume, rayHit0Coords + voxelVolumeSize/2).r >> 16 & 1) != 0) {
            uint hash = posToHash(rayHit0.xyz - 0.5 * rayNormal0);
            if ((atomicOr(lightHashMap[hash/32], 1<<hash%32) & uint(1)<<hash%32) == 0) {
                int lightIndex = atomicAdd(lightCount, 1);
                if (lightIndex < MAX_LIGHT_COUNT) {
                    positions[lightIndex] = ivec4(rayHit0Coords, 1);
                    vec3 lightPos = positions[lightIndex].xyz + 0.5;
                    float ndotl = rayHit0Coords == positions[lightIndex].xyz ? 1.0 :
                        max(0, dot(normalize(lightPos - vxPos), normalDepthData.xyz));
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
            int packedLightSubPos = imageLoad(
                voxelCols,
                (lightPos0 + voxelVolumeSize/2) * ivec3(1, 2, 1) + ivec3(0, 2 * voxelVolumeSize.y, 0)).r;
            int packedLightPosStdev = imageLoad(
                voxelCols,
                (lightPos0 + voxelVolumeSize/2) * ivec3(1, 2, 1) + ivec3(0, 2 * voxelVolumeSize.y, 0) + ivec3(0, 1, 0)).r;
            vec3 subLightPos = 0.1 * vec3(packedLightSubPos & 0x3ff, packedLightSubPos>>10 & 0x3ff, packedLightSubPos>>20 & 0x3ff) / (packedLightPosStdev>>13) - 1;
            ivec3 offsetCandidateBlock = ivec3(greaterThan(subLightPos, vec3(0.8)));
            bool hasGreaterLight = false;
            for (int i = 0; i < 3; i++) {
                if (
                    offsetCandidateBlock[i] == 1 &&
                    (imageLoad(occupancyVolume, lightPos0 + ivec3(equal(ivec3(i), ivec3(0, 1, 2))) + voxelVolumeSize/2).r >> 16 & 1) != 0) hasGreaterLight = true;
            }
            if (!hasGreaterLight) {
                uint hash = posToHash(lightPos0);
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
    }

    if (index < 4 * MAX_LIGHT_COUNT) {
        ivec2 offset = (index%4/2*2-1) * ivec2(index%2, (index+1)%2);
        int otherLightIndex = index / 4;
        ivec4 prevFrameLight = imageLoad(colorimg11, ivec2(gl_WorkGroupSize.xy) * (ivec2(gl_WorkGroupID.xy) + offset) + ivec2(otherLightIndex % gl_WorkGroupSize.x, otherLightIndex / gl_WorkGroupSize.x));
        prevFrameLight.xyz += vxPosFrameOffset;
        uint hash = posToHash(prevFrameLight.xyz);
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
                weights[thisLightIndex] = 0.0001 * prevFrameLight.w;
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
    }
    barrier();
    memoryBarrierShared();
    bool participateInSorting = index < MAX_LIGHT_COUNT/2;
    #include "/lib/misc/prepare4_BM_sort.glsl"

    if (index >= MAX_TRACE_COUNT) {
        if (index < (lightCount + 2 * MAX_TRACE_COUNT)/3) {
            vec3 lightPos = positions[index].xyz + 0.5;
            vec3 dir = lightPos - meanPos;
            float dirLen = length(dir);
            float ndotl = infnorm(lightPos - meanPos + 0.5 * meanNormal) < 0.5 ? 1.0 :
                mix(dot(normalize(lightPos - meanPos), meanNormal), 0.7, length(meanNormal));
            float totalBrightness = ndotl * (sqrt(1 - min(1.0, dirLen / LIGHT_TRACE_LENGTH))) / (dirLen + 0.1);
            int thisWeight = int(10000.5 * length(getColor(lightPos)) * totalBrightness);
            vec4 rayHit1 = coneTrace(meanPos, (1.0 - 0.1 / (dirLen + 0.1)) * dir, 0.3 / dirLen, dither);
            if (rayHit1.w > 0.01) positions[index].w = thisWeight;
        } else if (index < lightCount) {
            positions[index].w = 0;
        }
    }
    if (index < lightCount) {
        extraData[index] = imageLoad(occupancyVolume, positions[index].xyz + voxelVolumeSize/2).r;
    }
    barrier();
    memoryBarrierShared();

    vec3 writeColor = vec3(0);
    for (uint thisLightIndex = MAX_TRACE_COUNT * uint(!validData); thisLightIndex < min(lightCount, MAX_TRACE_COUNT); thisLightIndex++) {
        int packedLightSubPos = imageLoad(
            voxelCols,
            (positions[thisLightIndex].xyz + voxelVolumeSize/2) * ivec3(1, 2, 1) + ivec3(0, 2 * voxelVolumeSize.y, 0)).r;
        int packedLightPosStdev = imageLoad(
            voxelCols,
            (positions[thisLightIndex].xyz + voxelVolumeSize/2) * ivec3(1, 2, 1) + ivec3(0, 2 * voxelVolumeSize.y, 0) + ivec3(0, 1, 0)).r;
        vec3 subLightPos = 0.1 * vec3(packedLightSubPos & 0x3ff, packedLightSubPos>>10 & 0x3ff, packedLightSubPos>>20 & 0x3ff) / (packedLightPosStdev>>13) - 1;
        float lightSize = 0.5;//0.1 * (packedLightPosStdev & 0x1fff) / (packedLightPosStdev>>13);
        vec3 lightPos = positions[thisLightIndex].xyz + subLightPos;
        lightSize = clamp(lightSize, 0.01, getDistanceField(lightPos));
        float ndotl0 = dot(normalize(lightPos - vxPos), normalDepthData.xyz);
        ndotl0 = infnorm(vxPos - 0.1 * normalDepthData.xyz - lightPos) < 0.5 ? abs(ndotl0) : max(0, ndotl0);
        vec3 dir = lightPos - biasedVxPos;
        float dirLen = length(dir);
        float thisTraceLen = (extraData[thisLightIndex]>>17 & 31)/32.0;
        if (dirLen < thisTraceLen * LIGHT_TRACE_LENGTH && ndotl0 > 0.001) {
            float lightBrightness = 1.5 * thisTraceLen;
            lightBrightness *= lightBrightness;
            float ndotl = ndotl0 * lightBrightness;
            vec4 rayHit1 = coneTrace(biasedVxPos, (1.0 - 0.1 / (dirLen + 0.1)) * dir, lightSize * LIGHTSOURCE_SIZE_MULT / dirLen, dither);
            if (rayHit1.w > 0.01) {
                vec3 lightColor = getColor(positions[thisLightIndex].xyz + 0.5).xyz;
                float totalBrightness = ndotl * (sqrt(1 - dirLen / (LIGHT_TRACE_LENGTH * thisTraceLen))) / (dirLen + 0.1);
                writeColor += lightColor * rayHit1.w * totalBrightness;
                int thisWeight = int(10000.5 * length(lightColor) * totalBrightness);
                atomicMax(positions[thisLightIndex].w, thisWeight);
            }
        }
    }
    barrier();
    memoryBarrierShared();
    //if (index < lightCount) writeColor = vec3(10 * weights[index]);
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
