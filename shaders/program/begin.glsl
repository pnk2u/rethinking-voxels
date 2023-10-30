#include "/lib/common.glsl"

#ifdef CSH

#if VOXEL_DETAIL_AMOUNT <= 5
	const ivec3 workGroups = ivec3(16384, 1, 1);
#else
	const ivec3 workGroups = ivec3(15000, 1, 1);
#endif
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

uniform int frameCounter;

#define WRITE_TO_SSBOS
#define DECLARE_CAMPOS
#include "/lib/vx/SSBOs.glsl"

void main() {
	int mat = int(gl_WorkGroupID.x);
	if (!getMaterialAvailability(mat)) {
		return;
	}
	if (gl_LocalInvocationID.z == 0 && gl_LocalInvocationID.x < 7 && gl_LocalInvocationID.y == 0) {
		blockIdMap[workGroups.x+7*mat+gl_LocalInvocationID.x] = 0;
	}
	int baseIndex = getBaseIndex(mat);
	const int lodSubdivisions = 1<<(VOXEL_DETAIL_AMOUNT-1);
	const int workGroupSizeFactor = max(1, lodSubdivisions/8);
	const int invocationCount = workGroupSizeFactor * workGroupSizeFactor * workGroupSizeFactor;

	for (int invocation = 0; invocation < invocationCount; invocation++) {
		ivec3 coords = ivec3(invocation % workGroupSizeFactor,
								invocation / workGroupSizeFactor % workGroupSizeFactor,
								invocation / (workGroupSizeFactor * workGroupSizeFactor) % workGroupSizeFactor) * 8;
		coords += ivec3(gl_LocalInvocationID);
		if (any(greaterThan(coords, ivec3(lodSubdivisions-1)))) {
			continue;
		}
		int index = baseIndex + lodSubdivisions * (lodSubdivisions * coords.x + coords.y) + coords.z;
		geometryData[index] = 0;
	}
}
#endif

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

#define WRITE_TO_SSBOS
#define DECLARE_CAMPOS
#include "/lib/vx/SSBOs.glsl"

void main() {
	imageStore(voxelVolumeI, ivec3(gl_GlobalInvocationID), ivec4(0));
	imageStore(voxelVolumeI, ivec3(gl_GlobalInvocationID) + ivec3(0, voxelVolumeSize.y, 0), ivec4(0));
	imageStore(voxelVolumeI, ivec3(gl_GlobalInvocationID) + ivec3(0, 3 * voxelVolumeSize.y, 0), ivec4(0));
}
#endif