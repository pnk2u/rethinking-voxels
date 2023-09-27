#include "/lib/common.glsl"
// workGroups.x = 16384 + 512
const ivec3 workGroups = ivec3(16896, 1, 1);
layout(local_size_x=32, local_size_y = 1, local_size_z = 1) in;

uniform sampler2D colortex3;

#define DECLARE_CAMPOS
#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

void main() {
	if (gl_WorkGroupID.x < 512) {
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
	} else {
		int mat = int(gl_WorkGroupID.x);
		const int memSize = modelMemorySize + (maxEmissiveVoxels + 2) / 3 + 1;
		for (uint k = gl_LocalInvocationID.x; k < memSize; k += gl_WorkGroupSize.x) {
			geometryData[mat * memSize + k] = 0;
		}
	}
}