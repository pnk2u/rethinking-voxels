#include "/lib/common.glsl"
#if VX_VOL_SIZE == 0
    const ivec3 workGroups = ivec3(12, 8, 12);
#elif VX_VOL_SIZE == 1
    const ivec3 workGroups = ivec3(16, 12, 16);
#endif
//const ivec3 workGroups = ivec3(32, 16, 32);

layout(local_size_x = 10, local_size_y = 10, local_size_z = 10) in;

layout(rgba16f) uniform image3D distanceFieldI;
layout(r32i) readonly uniform iimage3D occupancyVolume;
layout(r32i) uniform iimage3D voxelCols;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform int frameCounter;

shared float fullDist[10][10][10];

void main() {
    ivec3 camOffset = ivec3(1.01 * (floor(cameraPosition) - floor(previousCameraPosition)));
    ivec3 baseCoord = ivec3(gl_WorkGroupID) * 8;
    ivec3 localCoord = ivec3(gl_LocalInvocationID) - 1;
    ivec3 texCoord = baseCoord + localCoord;
    vec4 theseDists = vec4(1000);
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
    if (all(greaterThanEqual(localCoord, ivec3(0))) && all(lessThan(localCoord, ivec3(8)))) {
        imageStore(distanceFieldI, texCoord + ivec3(0, (frameCounter+1)%2 * voxelVolumeSize.y, 0), theseDists);
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
    }
}