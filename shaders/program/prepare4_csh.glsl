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
#ifdef SCREENSPACE_LIGHT_DISCOVERY
    layout(rgba16i) uniform iimage2D colorimg11;
#else
    layout(rgba16i) uniform iimage3D lightStorage;
#endif

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
        #ifdef SCREENSPACE_LIGHT_DISCOVERY
            lightCount = 0;
        #else
            lightCount = 1000;
        #endif
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
    #ifdef SCREENSPACE_LIGHT_DISCOVERY
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
    #endif
    if (validData) {
        vec4 playerPos = gbufferModelViewInverse * (gbufferProjectionInverse * (vec4((readTexelCoord + 0.5) / view, 1 - normalDepthData.a, 1) * 2 - 1));
        playerPos /= playerPos.w;
        vxPos = playerPos.xyz + fract(cameraPosition);
        biasedVxPos = vxPos + max(1.5/(1<<VOXEL_DETAIL_AMOUNT), 2.5 * infnorm(playerPos.xyz/voxelVolumeSize)) * normalDepthData.xyz;
        vxPos = biasedVxPos;
        lightStorageCoords = ivec3(biasedVxPos + voxelVolumeSize/2)/8*8;
        #ifdef SCREENSPACE_LIGHT_DISCOVERY
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
        #endif
    }
    #ifdef SCREENSPACE_LIGHT_DISCOVERY
        vec3 meanPos = vec3(cumulatedPos.xyz)/cumulatedPos.w;
        vec3 meanNormal = vec3(cumulatedNormal.xyz)/cumulatedNormal.w;
        meanNormal *= length(meanNormal);

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
        barrier();
    #endif

    vec3 writeColor = vec3(0);
    for (uint thisLightIndex = MAX_TRACE_COUNT * uint(!validData); thisLightIndex < min(lightCount, MAX_TRACE_COUNT); thisLightIndex++) {
        #ifndef SCREENSPACE_LIGHT_DISCOVERY
            ivec4 lightPos0 = imageLoad(
                lightStorage,
                lightStorageCoords + ivec3(thisLightIndex%8, thisLightIndex/8%8, thisLightIndex/64%8)
            );
            if (lightPos0.w <= 0) continue;
            if ((imageLoad(occupancyVolume, lightPos0.xyz + voxelVolumeSize/2).r >> 16 & 1) == 0) continue;
            vec3 lightPos = lightPos0.xyz + 0.5;
        #else
            vec3 lightPos = positions[thisLightIndex].xyz + 0.5;
        #endif
        float ndotl0 = infnorm(vxPos - 0.5 * normalDepthData.xyz - lightPos) < 0.5 ? 1.0 : max(0, dot(normalize(lightPos - vxPos), normalDepthData.xyz));
        vec3 dir = lightPos - biasedVxPos;
        float dirLen = length(dir);
        if (dirLen < LIGHT_TRACE_LENGTH && ndotl0 > 0.001) {
            float lightBrightness = 1;//getLightLevel(ivec3(lightPos + 1000) - 1000 + voxelVolumeSize/2) * 0.04;
            lightBrightness *= lightBrightness;
            float ndotl = ndotl0 * lightBrightness;
            vec4 rayHit1 = coneTrace(biasedVxPos, (1.0 - 0.1 / (dirLen + 0.1)) * dir, 0.3 / dirLen, dither);
            if (rayHit1.w > 0.01) {
                vec3 lightColor = getColor(lightPos).xyz;
                float totalBrightness = ndotl * (sqrt(1 - dirLen / LIGHT_TRACE_LENGTH)) / (dirLen + 0.1);
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
    #ifdef SCREENSPACE_LIGHT_DISCOVERY
        ivec4 lightPosToStore = (index < lightCount && positions[index].w > 0) ? positions[index] : ivec4(0);
        imageStore(colorimg11, writeTexelCoord, lightPosToStore);
    #endif
}
#else
    const ivec3 workGroups = ivec3(1, 1, 1);
    layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
    void main() {}
#endif
#endif
