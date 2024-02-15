#include "/lib/common.glsl"
#if VX_VOL_SIZE == 0
    const ivec3 workGroups = ivec3(12, 8, 12);
#elif VX_VOL_SIZE == 1
    const ivec3 workGroups = ivec3(16, 12, 16);
#elif VX_VOL_SIZE == 2
    const ivec3 workGroups = ivec3(32, 16, 32);
#elif VX_VOL_SIZE == 3
    const ivec3 workGroups = ivec3(64, 16, 64);
#endif

layout(local_size_x = 10, local_size_y = 10, local_size_z = 10) in;

layout(rgba16f) uniform image3D distanceFieldI;
layout(r32i) uniform restrict iimage3D occupancyVolume;
layout(r32i) uniform restrict iimage3D voxelCols;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform int frameCounter;

bvec2 or(bvec2 a, bvec2 b) {
    return bvec2(a.x || b.x, a.y || b.y);
}

bvec3 or(bvec3 a, bvec3 b) {
    return bvec3(a.x || b.x, a.y || b.y, a.z || b.z);
}

bvec4 or(bvec4 a, bvec4 b) {
    return bvec4(a.x || b.x, a.y || b.y, a.z || b.z, a.w || b.w);
}

shared float fullDist[10][10][10];

void main() {
    ivec3 camOffset = ivec3(1.01 * (floor(cameraPosition) - floor(previousCameraPosition)));
    ivec3 baseCoord = ivec3(gl_WorkGroupID) * 8;
    ivec3 localCoord = ivec3(gl_LocalInvocationID) - 1;
    ivec3 texCoord = baseCoord + localCoord;
    float[8] theseDists;
    for (int k = 0; k < 8; k++) theseDists[k] = 1000;
    int thisOccupancy = imageLoad(occupancyVolume, texCoord).r;
    #define j 0
    #include "/program/shadowcomp_sdf_loop.glsl"
    #undef j
    #if VOXEL_DETAIL_AMOUNT > 1
        #define j 1
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 2
        #define j 2
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 3
        #define j 3
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 4
        #define j 4
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 5
        #define j 5
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 6
        #define j 6
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    #if VOXEL_DETAIL_AMOUNT > 7
        #define j 7
        #include "/program/shadowcomp_sdf_loop.glsl"
        #undef j
    #endif
    if (all(greaterThanEqual(localCoord, ivec3(0))) && all(lessThan(localCoord, ivec3(8)))) {
        imageStore(
            distanceFieldI,
            texCoord + ivec3(0, (frameCounter+1)%2 * 2 * voxelVolumeSize.y, 0),
            vec4(theseDists[0], theseDists[1], theseDists[2], theseDists[3]));
        imageStore(
            distanceFieldI,
            texCoord + ivec3(0, ((frameCounter+1)%2 * 2 + 1) * voxelVolumeSize.y, 0),
            vec4(theseDists[4], theseDists[5], theseDists[6], theseDists[7]));

        ivec2 rawCol = ivec2(
            imageLoad(voxelCols, texCoord * ivec3(1, 2, 1)).r,
            imageLoad(voxelCols, texCoord * ivec3(1, 2, 1) + ivec3(0, 1, 0)).r
        );
        if ((rawCol.g >> 23) == 0) {
            for (int k = 0; k < 6; k++) {
                ivec3 offset = (k/3*2-1) * ivec3(equal(ivec3(k%3), ivec3(0, 1, 2)));
                ivec2 otherRawCol = ivec2(
                    imageLoad(voxelCols, (texCoord + offset) * ivec3(1, 2, 1)).r,
                    imageLoad(voxelCols, (texCoord + offset) * ivec3(1, 2, 1) + ivec3(0, 1, 0)).r
                );
                if ((otherRawCol.g >> 23) > (rawCol.g >> 23)) {
                    rawCol = otherRawCol;
                    rawCol.g &= ~(0x3ff << 13);
                }
            }
            if ((rawCol.g >> 23) > 0) {
                imageStore(voxelCols, texCoord * ivec3(1, 2, 1), ivec4(rawCol.r));
                imageStore(voxelCols, texCoord * ivec3(1, 2, 1) + ivec3(0, 1, 0), ivec4(rawCol.g));
            }
        }
        if ((thisOccupancy >> 16 & 1) != 0) {
            ivec2 rawLightPos = ivec2(
                imageLoad(voxelCols, texCoord * ivec3(1, 2, 1) + ivec3(0, 2 * voxelVolumeSize.y, 0)).r,
                imageLoad(voxelCols, texCoord * ivec3(1, 2, 1) + ivec3(0, 1, 0) + ivec3(0, 2 * voxelVolumeSize.y, 0)).r
            );
            vec3 relLightPos = 0.1 * vec3(rawLightPos.x & 0x3ff, rawLightPos.x >> 10 & 0x3ff, rawLightPos.x >> 20 & 0x3ff) / (rawLightPos.y >> 13);
            if (any(greaterThan(relLightPos, vec3(0.6)))) {
                for (int k = 1; k < 8; k++) {
                    ivec3 offset = ivec3(k%2, k/2%2, k/4%2);
                    if ((imageLoad(occupancyVolume, texCoord + offset).r >> 16 & 1) != 0) {
                        ivec2 otherRawPos = ivec2(
                            imageLoad(voxelCols, (texCoord + offset) * ivec3(1, 2, 1) + ivec3(0, 2 * voxelVolumeSize.y, 0)).r,
                            imageLoad(voxelCols, (texCoord + offset) * ivec3(1, 2, 1) + ivec3(0, 1, 0) + ivec3(0, 2 * voxelVolumeSize.y, 0)).r
                        );
                        vec3 otherRelLightPos = 0.1 * vec3(otherRawPos.x & 0x3ff, otherRawPos.x >> 10 & 0x3ff, otherRawPos.x >> 20 & 0x3ff) / (otherRawPos.y >> 13);
                        if (length(offset + otherRelLightPos - relLightPos) < 0.8) {
                            imageAtomicAnd(occupancyVolume, texCoord, ~(1<<16));
                            break;
                        }
                    }
                }
            }
        }
    }
}