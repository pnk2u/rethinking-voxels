#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(512, 1, 1);
layout(local_size_x=32, local_size_y = 1, local_size_z = 1) in;

uniform sampler2D colortex3;

#define DECLARE_CAMPOS
#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

void main() {
    int mat = int(gl_GlobalInvocationID.x);
    if (mat / 512 < textureSize(colortex3, 0).y) {
        ivec2 mappedMat0 = ivec2(texelFetch(colortex3, ivec2(mat % 512, mat / 512), 0).rg * 255 + 0.5);
        blockIdMap[mat] = mappedMat0.x + (mappedMat0.y << 8);
    } else {
        blockIdMap[mat] = 0;
    }
    for (int k = 0; k < 7; k++) {
        blockIdMap[16384 + 7 * mat + k] = 0;
    }
}