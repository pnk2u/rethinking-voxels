#include "/lib/common.glsl"

//////1st Compute Shader//////1st Compute Shader//////1st Compute Shader//////
/*
This program offsets irradiance cache data to account for camera movement, and handles its temporal accumulation falloff
*/
#ifdef CSH

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

#include "/lib/vx/irradianceCache.glsl"

#define IRRADIANCECACHE_FALLOFF (1.0 - 0.1 * ACCUM_FALLOFF_SPEED)

void main() {
    ivec3 camOffset = ivec3(1.01 * (floor(cameraPosition) - floor(previousCameraPosition)));
    // this actually works for having threads be executed in the correct order so that they don't read the output of other previously run threads.
    ivec3 coords = ivec3(gl_GlobalInvocationID);
    coords = coords * ivec3(greaterThan(camOffset, ivec3(-1))) +
        (voxelVolumeSize - coords - 1) * ivec3(lessThan(camOffset, ivec3(0)));
    ivec3 prevCoords = coords + camOffset;
    vec4[2] writeColors;
    for (int k = 0; k < 2; k++) {
        writeColors[k] = (all(lessThan(prevCoords, voxelVolumeSize)) && all(greaterThanEqual(prevCoords, ivec3(0)))) ? imageLoad(irradianceCacheI, prevCoords + ivec3(0, k * voxelVolumeSize.y, 0)) : vec4(0);
    }
    barrier();
    memoryBarrierImage();
    for (int k = 0; k < 2; k++) {
        imageStore(irradianceCacheI, coords + ivec3(0, k * voxelVolumeSize.y, 0), writeColors[k] * IRRADIANCECACHE_FALLOFF);
    }
}
#endif

//////2nd Compute Shader//////2nd Compute Shader//////2nd Compute Shader//////
/*
this program calculates volumetric block lighting
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
#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/irradianceCache.glsl"
#include "/lib/vx/voxelReading.glsl"
#include "/lib/util/random.glsl"

#define MAX_LIGHT_COUNT 48

shared int lightCount;
shared bool anyInFrustrum;
shared ivec4[MAX_LIGHT_COUNT] positions;
shared int[MAX_LIGHT_COUNT] mergeOffsets;

shared vec3[5] frustrumSides;

const vec2[4] squareCorners = vec2[4](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, 1));

void main() {
    int index = int(gl_LocalInvocationID.x + gl_WorkGroupSize.x * (gl_LocalInvocationID.y + gl_WorkGroupSize.y * gl_LocalInvocationID.z));
    float dither = fract((10000 * frameCounter + dot(vec2(gl_GlobalInvocationID.xy), vec2(985, 173))) / 1.61803399);
    if (index < 4) {
        vec4 pos = vec4(squareCorners[index], 0.9999, 1);
        pos = gbufferModelViewInverse * (gbufferProjectionInverse * pos);
        frustrumSides[index] = pos.xyz * pos.w;
    } else if (index == 4) {
        frustrumSides[4] = -normalize(gbufferModelViewInverse[2].xyz);
        lightCount = 0;
        anyInFrustrum = false;
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
    ivec3 vxPosFrameOffset = ivec3((floor(previousCameraPosition) - floor(cameraPosition)) * 1.1);
    ivec3 coords = ivec3(gl_GlobalInvocationID);
    coords = coords * ivec3(lessThan(vxPosFrameOffset, ivec3(1))) +
        (voxelVolumeSize - coords - 1) * ivec3(greaterThan(vxPosFrameOffset, ivec3(0)));
    vec3 vxPos = coords - 0.5 * voxelVolumeSize + vec3(nextFloat(), nextFloat(), nextFloat()) * 0.6 + 0.2;
    bool insideFrustrum = true;
    for (int k = 0; k < 5; k++) {
        insideFrustrum = (insideFrustrum && dot(vxPos, frustrumSides[k]) > -10.0);
    }
    vec4 writeColor = vec4(0);
    vec3 normal = vec3(0);
    bool hasNeighbor = false;
    if (insideFrustrum) {
        anyInFrustrum = true;
        writeColor = imageLoad(irradianceCacheI, coords + ivec3(0, voxelVolumeSize.y, 0));
        #ifdef GI
            vec4 GILight = imageLoad(irradianceCacheI, coords);
        #endif
        vec3 absNormal = vec3(0);
        for (int k = 0; k < 27; k++) {
            ivec3 offset = ivec3(k%3, k/3%3, k/9%3) - 1;
            if (offset != ivec3(0) && (imageLoad(occupancyVolume, coords + offset).r & 1) > 0) {
                hasNeighbor = true;
                normal -= offset;
                absNormal += abs(offset);
            }
        }
        if (hasNeighbor) {
            normal = all(lessThan(abs(normal), vec3(0.01))) ? absNormal : normal;
            normal = normalize(normal);
        }
        if ((imageLoad(occupancyVolume, coords).r & 17) > 0) {
            int lightIndex = atomicAdd(lightCount, 1);
            if (lightIndex < MAX_LIGHT_COUNT) {
                positions[lightIndex] = ivec4(coords - voxelVolumeSize / 2, 0);
                mergeOffsets[lightIndex] = 0;
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
        vec3 dir = LIGHT_TRACE_LENGTH * randomSphereSample();
        float ndotl = dot(dir, normal) / LIGHT_TRACE_LENGTH;
        if (hasNeighbor && ndotl < 0) {
            dir *= -1;
            ndotl *= -1;
        }
        vec3 rayNormal0;
        vec4 rayHit0 = voxelTrace(vxPos, dir, rayNormal0);

        if (rayHit0.a > 16) {
            int lightIndex = atomicAdd(lightCount, 1);
            if (lightIndex < MAX_LIGHT_COUNT) {
                ivec3 lightPos = ivec3(rayHit0.xyz - 0.01 * rayNormal0 + 1000) - 1000;
                positions[lightIndex] = ivec4(lightPos, 0);
                mergeOffsets[lightIndex] = 0;
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
        #ifdef GI
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
    }
    barrier();
    memoryBarrierShared();
    if (anyInFrustrum && index < 6) {
        ivec3 offset = ivec3(equal(ivec3(index % 3), ivec3(0, 1, 2))) * (index / 3 * 2 - 1);
        int otherLightIndex = frameCounter % MAX_LIGHT_COUNT % (lightCount * 2 + 2);
        ivec3 aroundLight = imageLoad(occupancyVolume, ivec3(gl_WorkGroupSize * (gl_WorkGroupID + offset)) + ivec3(otherLightIndex % gl_WorkGroupSize.x, otherLightIndex / gl_WorkGroupSize.x % gl_WorkGroupSize.y, otherLightIndex / (gl_WorkGroupSize.x * gl_WorkGroupSize.y) % gl_WorkGroupSize.z) + ivec3(0, voxelVolumeSize.y, 0)).xxx;
        bool isCurrentLight = false;
        if (aroundLight.x != 0) {
            isCurrentLight = (aroundLight.x / (voxelVolumeSize.x * voxelVolumeSize.y * voxelVolumeSize.z) == frameCounter % 2);
            aroundLight = ivec3(
                aroundLight.x % voxelVolumeSize.x,
                aroundLight.x / voxelVolumeSize.x % voxelVolumeSize.y,
                aroundLight.x / (voxelVolumeSize.x * voxelVolumeSize.y) % voxelVolumeSize.z
            ) - voxelVolumeSize / 2;
        }
        bool known = (aroundLight == ivec3(0) || (imageLoad(occupancyVolume, aroundLight + voxelVolumeSize/2).r & 1) == 0);
        if (!isCurrentLight) {
            aroundLight += vxPosFrameOffset;
        }
        if (!known) {
            int thisLightIndex = atomicAdd(lightCount, 1);
            if (thisLightIndex < MAX_LIGHT_COUNT) {
                positions[thisLightIndex] = ivec4(aroundLight, 0);
                mergeOffsets[thisLightIndex] = 0;
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
    }
    barrier();
    memoryBarrierShared();
    int oldLightCount = lightCount;
    int mergeOffset = 0;
    ivec4 thisPos = ivec4(0);
    int k = index + 1;
    if (index < oldLightCount) {
        thisPos = positions[index];
        while (k < oldLightCount && positions[k].xyz != thisPos.xyz) k++;
        if (k < oldLightCount) {
            atomicAdd(mergeOffsets[k], -1000);
            mergeOffset = 1;
            for (k++; k < oldLightCount; k++) {
                atomicAdd(mergeOffsets[k], 1);
            }
        }
    }
    barrier();
    memoryBarrierShared();
    if (index < oldLightCount) {
        if (mergeOffsets[index] > 0) {
            positions[index - mergeOffsets[index]] = thisPos;
        }
        if (mergeOffset > 0) {
            atomicAdd(lightCount, -1);
            atomicMax(lightCount, 0);
        }
    }
    barrier();
    memoryBarrierShared();
    oldLightCount = lightCount;
    barrier();
    memoryBarrierShared();
    if (anyInFrustrum && index < MAX_LIGHT_COUNT) {
        ivec3 prevFrameLight = imageLoad(occupancyVolume, coords + ivec3(0, voxelVolumeSize.y, 0)).xxx;
        if (prevFrameLight.x != 0) {
            prevFrameLight = ivec3(
                prevFrameLight.x % voxelVolumeSize.x,
                prevFrameLight.x / voxelVolumeSize.x % voxelVolumeSize.y,
                prevFrameLight.x / (voxelVolumeSize.x * voxelVolumeSize.y) % voxelVolumeSize.z
            ) - voxelVolumeSize / 2;
            prevFrameLight += vxPosFrameOffset;
        }
        bool known = (prevFrameLight == ivec3(0) || (imageLoad(occupancyVolume, prevFrameLight + voxelVolumeSize/2).r & 1) == 0);
        for (int k = 0; k < oldLightCount; k++) {
            if (prevFrameLight == positions[k].xyz) {
                known = true;
                break;
            }
        }
        if (!known) {
            int thisLightIndex = atomicAdd(lightCount, 1);
            if (thisLightIndex < MAX_LIGHT_COUNT) {
                positions[thisLightIndex] = ivec4(prevFrameLight, 0);
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
    }
    barrier();
    memoryBarrierShared();
    if (insideFrustrum) {
        vec3 newLightColor = vec3(0);
        #if defined TRACE_ALL_LIGHTS
            for (uint thisLightIndex = 0; thisLightIndex < min(lightCount, MAX_LIGHT_COUNT); thisLightIndex++) {
        #else
            if (lightCount > 0) {
                uint thisLightIndex = nextUint() % min(lightCount, MAX_LIGHT_COUNT);
            #endif
            vec3 lightPos = vec3(positions[thisLightIndex].xyz);
            ivec3 lightCoords = ivec3(lightPos + 1000) - 1000 + voxelVolumeSize / 2;
            int lightData = imageLoad(occupancyVolume, lightCoords).r;

            if ((lightData & 16) == 0) {
                lightPos = vec3(-10000);
            }
            vec3 dir = lightPos - vxPos;
            float dirLen = length(dir);
            float ndotl = max(0, hasNeighbor ? max(0, dot(normalize(dir + normal), normal)) : 1.0);
            if (dirLen < LIGHT_TRACE_LENGTH && ndotl > 0.001) {
                float lightBrightness = getLightLevel(positions[thisLightIndex].xyz + voxelVolumeSize/2) * 0.04;
                lightBrightness *= lightBrightness;
                vec4 rayHit1 = coneTrace(vxPos, (1.0 + 0.1 / (dirLen + 0.1)) * dir, 0.3 / dirLen, dither);
                vec3 rayNormal1 = normalize(distanceFieldGradient(rayHit1.xyz));
                if (rayHit1.a > 0.01 && infnorm(rayHit1.xyz - 0.05 * rayNormal1 - positions[thisLightIndex].xyz - 0.5) < 0.51) {
                    newLightColor += getColor(rayHit1.xyz).rgb * ndotl * lightBrightness * sqrt(1.01 - dirLen / LIGHT_TRACE_LENGTH) / (dirLen + 0.1);
                    atomicAdd(positions[thisLightIndex].w, 1);
                }
            }
        }
        #ifndef TRACE_ALL_LIGHTS
            newLightColor *= lightCount;
        #endif
        writeColor += vec4(newLightColor, 1.0);
        writeColor *= 1.0 - max(0.1 / (lightCount * lightCount + 1) - 0.01, 0.0);
        imageStore(irradianceCacheI, coords + ivec3(0, voxelVolumeSize.y, 0), writeColor);
    }
    barrier();
    memoryBarrierShared();
    if (anyInFrustrum || vxPosFrameOffset != ivec3(0)) {
        imageStore(
            occupancyVolume,
            coords + ivec3(0, voxelVolumeSize.y, 0),
            index < lightCount && positions[index].w > 0 && isInRange(1.02 * positions[index].xyz) ?
            ivec4(
                positions[index].x + voxelVolumeSize.x / 2 +
                voxelVolumeSize.x * (positions[index].y + voxelVolumeSize.y / 2 +
                voxelVolumeSize.y * (positions[index].z + voxelVolumeSize.z / 2 +
                voxelVolumeSize.z * frameCounter % 2))
            ) : ivec4(0)
        );
    }
}
#endif
