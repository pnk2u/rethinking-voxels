#include "/lib/common.glsl"

#ifdef CSH

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

#define MAX_LIGHT_COUNT 48

shared int lightCount;
shared int finalLightCount;
shared ivec4[MAX_LIGHT_COUNT] positions;
shared int[MAX_LIGHT_COUNT] mergeOffsets;
shared uint[128] lightHashMap;

int packPosition(ivec3 pos) {
    pos += voxelVolumeSize/2;
    return pos.x + (pos.y << 9) + (pos.z << 18);
}
int packPosition(vec3 pos) {
    return packPosition(ivec3(pos + 1000) - 1000);
}
uint posToHash(int pos) {
    return 3591 * uint(pos) % uint(128*32);
}

void main() {
    int index = int(gl_LocalInvocationID.x + gl_WorkGroupSize.x * gl_LocalInvocationID.y);
    float dither = nextFloat();
    if (gl_LocalInvocationID.xy == uvec2(0)) {
        lightCount = 0;
    }
    if (index < 128) {
        lightHashMap[index] = 0;
    }
    barrier();
    memoryBarrierShared();
    ivec2 readTexelCoord = ivec2(gl_GlobalInvocationID.xy) * 2 + ivec2(frameCounter % 2, frameCounter / 2 % 2);
    ivec2 writeTexelCoord = ivec2(gl_GlobalInvocationID.xy);
    vec4 normalDepthData = texelFetch(colortex8, readTexelCoord, 0);
    ivec3 vxPosFrameOffset = ivec3((floor(previousCameraPosition) - floor(cameraPosition)) * 1.1);
    bool validData = (normalDepthData.a < 1.5 && length(normalDepthData.rgb) > 0.1 && all(lessThan(readTexelCoord, ivec2(view + 0.1))));
    vec3 vxPos = vec3(1000);
    if (validData) {
        vec4 playerPos = gbufferModelViewInverse * (gbufferProjectionInverse * (vec4((readTexelCoord + 0.5) / view, 1 - normalDepthData.a, 1) * 2 - 1));
        playerPos /= playerPos.w;
        vxPos = playerPos.xyz + fract(cameraPosition) + max(0.05, 1.2 * infnorm(playerPos.xyz/voxelVolumeSize)) * normalDepthData.xyz;
        vec3 dir = randomSphereSample();
        if (dot(dir, normalDepthData.xyz) < 0) dir *= -1;
        vec3 rayNormal0;
        vec4 rayHit0 = voxelTrace(vxPos, LIGHT_TRACE_LENGTH * dir, rayNormal0);
        if (rayHit0.a > 16) {
            int packedPos = packPosition(rayHit0.xyz - 0.05 * rayNormal0);
            uint hash = posToHash(packedPos);
            if ((atomicOr(lightHashMap[hash/32], 1<<hash%32) & 1<<hash%32) == 0) {
                int lightIndex = atomicAdd(lightCount, 1);
                if (lightIndex < MAX_LIGHT_COUNT) {
                    positions[lightIndex] = ivec4(rayHit0.xyz - 0.05 * rayNormal0 + 1000, 1) - ivec4(1000, 1000, 1000, 0);
                    mergeOffsets[lightIndex] = 0;
                } else {
                    atomicMin(lightCount, MAX_LIGHT_COUNT);
                }
            }
        }
    }
    if (index < 4 * MAX_LIGHT_COUNT) {
        ivec2 offset = (index%4/2*2-1) * ivec2(index%2, (index+1)%2);
        int otherLightIndex = index / 4;
        ivec4 prevFrameLight = imageLoad(colorimg11, ivec2(gl_WorkGroupSize.xy) * (ivec2(gl_WorkGroupID.xy) + offset) + ivec2(otherLightIndex % gl_WorkGroupSize.x, otherLightIndex / gl_WorkGroupSize.x));
        prevFrameLight.xyz += vxPosFrameOffset;
        uint hash = posToHash(packPosition(prevFrameLight.xyz));
        bool known = (
            prevFrameLight.w == 0 ||
            (imageLoad(occupancyVolume, prevFrameLight.xyz + voxelVolumeSize/2).r >> 16 & 1) == 0
        );
        if (!known) {
            known = (atomicOr(lightHashMap[hash/32], 1<<hash%32) & 1<<hash%32) != 0;
        }
        if (!known) {
            int thisLightIndex = atomicAdd(lightCount, 1);
            if (thisLightIndex < MAX_LIGHT_COUNT) {
                imageStore(colorimg10, writeTexelCoord, vec4(offset, -offset));
                positions[thisLightIndex] = ivec4(prevFrameLight.xyz, 0);
                mergeOffsets[thisLightIndex] = 0;
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
    }
    if (index < MAX_LIGHT_COUNT) {
        ivec4 prevFrameLight = imageLoad(colorimg11, writeTexelCoord);
        prevFrameLight.xyz += vxPosFrameOffset;

        uint hash = posToHash(packPosition(prevFrameLight.xyz));
        bool known = (
            prevFrameLight.w == 0 ||
            (imageLoad(occupancyVolume, prevFrameLight.xyz + voxelVolumeSize/2).r >> 16 & 1) == 0 ||
            (imageLoad(occupancyVolume, prevFrameLight.xyz + voxelVolumeSize/2).r >> 16 & 1) == 0
        );
        if (!known) {
            known = (atomicOr(lightHashMap[hash/32], 1<<hash%32) & 1<<hash%32) != 0;
        }

        if (!known) {
            int thisLightIndex = atomicAdd(lightCount, 1);
            if (thisLightIndex < MAX_LIGHT_COUNT) {
                positions[thisLightIndex] = ivec4(prevFrameLight.xyz, 0);
            } else {
                atomicMin(lightCount, MAX_LIGHT_COUNT);
            }
        }
    }
    barrier();
    memoryBarrierShared();
    vec3 writeColor = vec3(0);
    for (uint thisLightIndex = lightCount * uint(!validData); thisLightIndex < lightCount; thisLightIndex++) {
        vec3 lightPos = positions[thisLightIndex].xyz + 0.5;
        float ndotl0 = infnorm(vxPos - 0.5 * normalDepthData.xyz - lightPos) < 0.5 ? 1.0 : max(0, dot(normalize(lightPos - vxPos), normalDepthData.xyz));
        ivec3 lightCoords = ivec3(lightPos + 1000) - 1000 + voxelVolumeSize / 2;
        int lightData = imageLoad(occupancyVolume, lightCoords).r;
        if ((lightData & 1<<16) == 0) {
            lightPos = vec3(-10000);
        }
        vec3 dir = lightPos - vxPos;
        float dirLen = length(dir);
        if (dirLen < LIGHT_TRACE_LENGTH) {
            float lightBrightness = 1;//getLightLevel(ivec3(lightPos + 1000) - 1000 + voxelVolumeSize/2) * 0.04;
            lightBrightness *= lightBrightness;
            float ndotl = ndotl0 * lightBrightness;
            vec4 rayHit1 = coneTrace(vxPos, (1.0 - 0.1 / (length(dir) + 0.1)) * dir, 0.3 / dirLen, dither);
            if (rayHit1.w > 0.01) {
                vec3 lightColor = getColor(lightPos).xyz;
                writeColor += lightColor * rayHit1.w * ndotl * (sqrt(1 - dirLen / LIGHT_TRACE_LENGTH)) / (dirLen + 0.1);
                atomicAdd(positions[thisLightIndex].w, 1);
            }
        }
    }
    #ifndef TRACE_ALL_LIGHTS
        writeColor *= lightCount;
    #endif
    if (gl_LocalInvocationID == uvec3(0)) {
        finalLightCount = 0;
    }
    barrier();
    memoryBarrierShared();
    if (index < lightCount && positions[index].w > 0) {
        atomicAdd(finalLightCount, 1);
    }
    barrier();
    memoryBarrierShared();

    imageStore(colorimg10, writeTexelCoord, vec4(writeColor, finalLightCount));
    barrier();
    memoryBarrierShared();
    ivec4 lightPosToStore = (index < lightCount && positions[index].w > 0) ? positions[index] : ivec4(0);
    imageStore(colorimg11, writeTexelCoord, lightPosToStore);
}
#endif
