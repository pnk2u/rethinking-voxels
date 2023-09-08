#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(511, 1, 1);
layout(local_size_x=32, local_size_y = 1, local_size_z = 1) in;

uniform sampler2D colortex3;

#define DECLARE_CAMPOS
#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

void main() {
    int mat = int(gl_GlobalInvocationID.x);
    ivec2 mappedMat0 = ivec2(texelFetch(colortex3, ivec2(mat % 512, mat / 512), 0).rg * 255 + 0.5);
    blockIdMap[mat] = mappedMat0.x + (mappedMat0.y << 8);
}