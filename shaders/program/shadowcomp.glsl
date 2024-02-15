#include "/lib/common.glsl"

//////1st Compute Shader//////1st Compute Shader//////1st Compute Shader//////
/*
This program offsets irradiance cache data to account for camera movement, and handles its temporal accumulation falloff
*/
#ifdef CSH_A

#if VX_VOL_SIZE == 0
    const ivec3 workGroups = ivec3(12, 8, 12);
#elif VX_VOL_SIZE == 1
    const ivec3 workGroups = ivec3(16, 12, 16);
#elif VX_VOL_SIZE == 2
    const ivec3 workGroups = ivec3(32, 16, 32);
#elif VX_VOL_SIZE == 3
    const ivec3 workGroups = ivec3(64, 16, 64);
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform int frameCounter;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

#define WRITE_TO_SSBOS

layout(rgba16f) uniform image3D irradianceCacheI;
layout(rgba16i) uniform iimage3D lightStorage;

void main() {
    ivec3 camOffset = ivec3(1.01 * (floor(cameraPosition) - floor(previousCameraPosition)));
    ivec3 coords = ivec3(gl_GlobalInvocationID);
    // this actually works for having threads be executed in the correct order so that they don't read the output of other previously run threads, so I should uncomment it if I need this program to offset data again
   coords = coords * ivec3(greaterThan(camOffset, ivec3(-1))) +
        (voxelVolumeSize - coords - 1) * ivec3(lessThan(camOffset, ivec3(0)));
    ivec4 lightPos = imageLoad(lightStorage, coords);
    ivec3 prevCoords = coords + camOffset;
    vec4[2] writeColors;
    for (int k = 0; k < 2; k++) {
        writeColors[k] = (all(lessThan(prevCoords, voxelVolumeSize)) && all(greaterThanEqual(prevCoords, ivec3(0)))) ? imageLoad(irradianceCacheI, prevCoords + ivec3(0, k * voxelVolumeSize.y, 0)) : vec4(0);
    }
    barrier();
    memoryBarrierImage();
    for (int k = 0; k < 2; k++) {
        imageStore(irradianceCacheI, coords + ivec3(0, k * voxelVolumeSize.y, 0), writeColors[k]);
    }
    imageStore(lightStorage, coords, lightPos - ivec4(camOffset, 0));
}
#endif

//////2nd Compute Shader//////2nd Compute Shader//////2nd Compute Shader//////
/*
this program calculates volumetric block lighting
*/
#ifdef CSH_B
#if VX_VOL_SIZE == 0
    const ivec3 workGroups = ivec3(12, 8, 12);
#elif VX_VOL_SIZE == 1
    const ivec3 workGroups = ivec3(16, 12, 16);
#elif VX_VOL_SIZE == 2
    const ivec3 workGroups = ivec3(32, 16, 32);
#elif VX_VOL_SIZE == 3
    const ivec3 workGroups = ivec3(64, 16, 64);
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform int frameCounter;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
#if defined REALTIME_SHADOWS && defined GI && defined OVERWORLD
    uniform mat4 gbufferModelView;
    uniform mat4 shadowModelView;
    uniform mat4 shadowProjection;
    uniform vec3 skyColor;
    uniform ivec2 eyeBrightness;

    const vec2 sunRotationData = vec2(cos(sunPathRotation * 0.01745329251994), -sin(sunPathRotation * 0.01745329251994));
    float ang = (fract(timeAngle - 0.25) + (cos(fract(timeAngle - 0.25) * 3.14159265358979) * -0.5 + 0.5 - fract(timeAngle - 0.25)) / 3.0) * 6.28318530717959;
    vec3 sunVec = vec3(-sin(ang), cos(ang) * sunRotationData);
    vec3 lightVec = sunVec * ((timeAngle < 0.5325 || timeAngle > 0.9675) ? 1.0 : -1.0);
    float SdotU = sunVec.y;
    float sunFactor = SdotU < 0.0 ? clamp(SdotU + 0.375, 0.0, 0.75) / 0.75 : clamp(SdotU + 0.03125, 0.0, 0.0625) / 0.0625;
    float sunVisibility = clamp(SdotU + 0.0625, 0.0, 0.125) / 0.125;
    float sunVisibility2 = sunVisibility * sunVisibility;

    #define NOT_IN_FRAGMENT
    #include "/lib/util/spaceConversion.glsl"
    #include "/lib/lighting/shadowSampling.glsl"
    #include "/lib/colors/lightAndAmbientColors.glsl"
#endif

layout(rgba16f) uniform image3D irradianceCacheI;
layout(rgba16i) uniform iimage3D lightStorage;
#include "/lib/vx/voxelReading.glsl"
#include "/lib/util/random.glsl"

#if MAX_TRACE_COUNT < 128
    #define MAX_LIGHT_COUNT 128
#else
    #define MAX_LIGHT_COUNT 512
#endif
shared int lightCount;
shared bool anyInFrustrum;
shared ivec4[MAX_LIGHT_COUNT] positions;
shared float[MAX_LIGHT_COUNT] weights;
shared int[MAX_LIGHT_COUNT] mergeOffsets;
shared uint[128] lightHashMap;
shared vec3[5] frustrumSides;

const vec2[4] squareCorners = vec2[4](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1));
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
    int index = int(gl_LocalInvocationID.x + gl_WorkGroupSize.x * (gl_LocalInvocationID.y + gl_WorkGroupSize.y * gl_LocalInvocationID.z));
    float dither = nextFloat();
    if (index < 4) {
        vec4 pos = vec4(squareCorners[index], 0.9999, 1);
        pos = gbufferModelViewInverse * (gbufferProjectionInverse * pos);
        frustrumSides[index] = pos.xyz * pos.w;
    } else if (index == 4) {
        frustrumSides[4] = -normalize(gbufferModelViewInverse[2].xyz);
        lightCount = 0;
        anyInFrustrum = false;
    }
    if (index < 128) {
        lightHashMap[index] = 0;
    }
    barrier();
    memoryBarrierShared();
    vec3 sideNormal = vec3(0);
    if (index < 4) {
        sideNormal = -normalize(cross(frustrumSides[index], frustrumSides[(index+1)%4]));
    }
    barrier();
    if(index < 4) {
        frustrumSides[index] = sideNormal;
    }
    barrier();
    memoryBarrierShared();
    ivec3 coords = ivec3(gl_GlobalInvocationID);
    vec3 normal = vec3(0);
    vec3 vxPos = coords - 0.5 * voxelVolumeSize + vec3(0.51, 0.49, 0.502);
    bool insideFrustrum = true;
    for (int k = 0; k < 5; k++) {
        insideFrustrum = (insideFrustrum && dot(vxPos, frustrumSides[k]) > -10.0);
    }
    bool hasNeighbor = false;
    bool activeFrame = int(gl_WorkGroupID.x + gl_WorkGroupID.y + gl_WorkGroupID.z) % 10 == frameCounter * 3 % 10;

    if (insideFrustrum && activeFrame) {
        anyInFrustrum = true;
        hasNeighbor = getDistanceField(vxPos) < 0.7;
        normal = hasNeighbor ? distanceFieldGradient(vxPos) : vec3(0);
        if (hasNeighbor) {
            normal = normalize(normal);
            vxPos -= 0.3 * normal;
        }

        if ((imageLoad(occupancyVolume, coords).r & 1<<16) > 0) {
            uint hash = posToHash(coords - voxelVolumeSize/2);
            if ((atomicOr(lightHashMap[hash/32], uint(1)<<hash%32) & uint(1)<<hash%32) == 0) {
                int lightIndex = atomicAdd(lightCount, 1);
                if (lightIndex < MAX_LIGHT_COUNT) {
                    positions[lightIndex] = ivec4(coords - voxelVolumeSize / 2, 0);
                    weights[lightIndex] = length(getColor(positions[lightIndex].xyz + 0.5));
                } else {
                    atomicMin(lightCount, MAX_LIGHT_COUNT);
                }
            }
        }
    }

    barrier();
    memoryBarrierShared();
    if (index < MAX_LIGHT_COUNT && anyInFrustrum) {
        ivec4 prevFrameLight = imageLoad(lightStorage, coords);

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
    if (index < MAX_LIGHT_COUNT && anyInFrustrum) {
        for (int k = 0; k < 6; k++) {
            ivec3 offset = (k/3*2-1) * ivec3(equal(ivec3(k%3), ivec3(0, 1, 2)));
            ivec4 prevFrameLight = imageLoad(
                lightStorage,
                ivec3(gl_WorkGroupSize.xyz) * (ivec3(gl_WorkGroupID.xyz) + offset) +
                ivec3(
                    index % gl_WorkGroupSize.x,
                    index / gl_WorkGroupSize.x % gl_WorkGroupSize.y,
                    index / (gl_WorkGroupSize.x * gl_WorkGroupSize.y)));
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
    }
    barrier();
    memoryBarrierShared();
    bool participateInSorting = index < MAX_LIGHT_COUNT/2;
    #include "/lib/misc/prepare4_BM_sort.glsl"
    
    vec3 meanPos = vec3(gl_WorkGroupID) * 8 + 4 - 0.5 * voxelVolumeSize;
    if (false && index >= MAX_TRACE_COUNT && anyInFrustrum) {
        if (index < lightCount) {
            vec3 lightPos = positions[index].xyz + 0.5;
            vec3 dir = lightPos - meanPos;
            float dirLen = length(dir);
            float ndotl = 1.0;
            float totalBrightness = ndotl * (sqrt(1 - min(1.0, dirLen / LIGHT_TRACE_LENGTH))) / (dirLen + 0.1);
            int thisWeight = int(10000.5 * length(getColor(lightPos)) * totalBrightness);
            vec4 rayHit1 = coneTrace(meanPos, (1.0 - 0.1 / (dirLen + 0.1)) * dir, 0.3 / dirLen, dither);
            if (rayHit1.w > 0.01) positions[index].w = thisWeight;
        } else if (index < lightCount) {
            positions[index].w = 0;
        }
    }
    barrier();

    vec3 writeColor = vec3(0);
    for (uint thisLightIndex = MAX_LIGHT_COUNT * uint(!insideFrustrum || !activeFrame); thisLightIndex < min(lightCount, MAX_LIGHT_COUNT); thisLightIndex++) {
        vec3 lightPos = positions[thisLightIndex].xyz + 0.5;
        float ndotl0 = infnorm(vxPos - 0.5 * normal - lightPos) < 0.5 || !hasNeighbor ? 1.0 :
            max(0, (dot(normalize(lightPos - vxPos + 0.5 * normal), normal)));
        ivec3 lightCoords = positions[thisLightIndex].xyz + voxelVolumeSize / 2;
        vec3 dir = lightPos - vxPos;
        float dirLen = length(dir);
        if (dirLen < LIGHT_TRACE_LENGTH && ndotl0 > 0.001) {
            float lightBrightness = 1;//getLightLevel(ivec3(lightPos + 1000) - 1000 + voxelVolumeSize/2) * 0.04;
            lightBrightness *= lightBrightness;
            float ndotl = ndotl0 * lightBrightness;
            vec4 rayHit1 = coneTrace(vxPos, (1.0 - 0.1 / (dirLen + 0.1)) * dir, 0.4 / dirLen, dither);
            if (rayHit1.w > 0.01) {
                vec3 lightColor = getColor(lightPos).xyz;
                float totalBrightness = ndotl * (sqrt(1 - dirLen / LIGHT_TRACE_LENGTH)) / (dirLen + 0.1);
                writeColor += lightColor * rayHit1.w * totalBrightness;
                int thisWeight = int(10000.5 * length(lightColor) * totalBrightness);
                atomicMax(positions[thisLightIndex].w, thisWeight);
            }
        }
    }

    ivec4 thisLight;
    if (index < lightCount && anyInFrustrum) {
        thisLight = positions[index];
        mergeOffsets[index] = 0;
    }
    barrier();
    memoryBarrierShared();
    if (index < lightCount && anyInFrustrum && thisLight.w <= 0) {
        for (int j = index + 1; j < lightCount; j++) {
            atomicAdd(mergeOffsets[j], -1);
        }
    }
    barrier();
    if (index < lightCount && anyInFrustrum && thisLight.w > 0) {
        positions[index + mergeOffsets[index]] = thisLight;
    }
    barrier();
    if (lightCount > 0 && anyInFrustrum && index == 0) {
        lightCount += mergeOffsets[lightCount - 1];
    }
    barrier();
    memoryBarrierShared();

    if (anyInFrustrum) {
        imageStore(irradianceCacheI, coords + ivec3(0, voxelVolumeSize.y, 0), vec4(writeColor, 1));
        ivec4 lightPosToStore = (index < lightCount && positions[index].w > 0) ? positions[index] : ivec4(0);
        imageStore(lightStorage, coords, lightPosToStore);
    }
}
/*        #ifdef GI
            else if (hasNeighbor) {
                if (rayHit0.rayColor.a > 0.999999) {
                    ivec3 hitCoords = ivec3(rayHit0.pos + 0.1 * rayHit0.normal + 0.5 * voxelVolumeSize);
                    vec4 blocklightHere = imageLoad(irradianceCacheI, hitCoords + ivec3(0, voxelVolumeSize.y, 0));
                    vec4 bouncedLightHere = imageLoad(irradianceCacheI, hitCoords);
                    #if defined REALTIME_SHADOWS && defined OVERWORLD
                        vec3 playerPos = rayHit0.pos - fract(cameraPosition);
                        float distanceBias = pow(dot(playerPos, playerPos), 0.75);
                        distanceBias = 0.12 + 0.0008 * distanceBias;
                        playerPos += distanceBias * rayHit0.normal;
                        vec3 shadowPos = GetShadowPos(playerPos);
                        vec3 shadow = SampleShadow(shadowPos, 1, 1) * clamp(pow2(pow2(eyeBrightness.y / 60.0)), 0.0, 1.0);
                        vec3 totalLight = shadow * lightColor * max(0, dot(rayHit0.normal, lightVec));
                    #else
                        vec3 totalLight = vec3(0);
                    #endif
                    totalLight += blocklightHere.xyz / max(blocklightHere.a, 0.0001) + bouncedLightHere.xyz / max(bouncedLightHere.a, 0.0001);
                    vec3 lightMult = pow2(rayHit0.rayColor.rgb) / max(max(max(rayHit0.rayColor.r, rayHit0.rayColor.g), rayHit0.rayColor.b), 0.01) * ndotl;
                    GILight += vec4(totalLight * lightMult, 1);
                } else {
                    GILight.a += 1;
                }
            }
            imageStore(irradianceCacheI, coords, GILight);
        #endif
        */
#endif

// This program calculates GI
#ifdef CSH_C
#if VX_VOL_SIZE == 0
    const ivec3 workGroups = ivec3(12, 8, 12);
#elif VX_VOL_SIZE == 1
    const ivec3 workGroups = ivec3(16, 12, 16);
#elif VX_VOL_SIZE == 2
    const ivec3 workGroups = ivec3(32, 16, 32);
#elif VX_VOL_SIZE == 3
    const ivec3 workGroups = ivec3(64, 16, 64);
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

#endif