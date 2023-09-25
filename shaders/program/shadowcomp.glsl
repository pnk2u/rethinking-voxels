#include "/lib/common.glsl"
//////1st Compute Shader//////1st Compute Shader//////1st Compute Shader//////
#ifdef CSH

const ivec3 workGroups = ivec3(16384, 1, 1);
#if VOXEL_DETAIL_AMOUNT == 1
	layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
#elif VOXEL_DETAIL_AMOUNT == 2
	layout(local_size_x = 2, local_size_y = 2, local_size_z = 2) in;
#else
	layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
#endif
shared ivec4 emissiveParts[64];
shared int sortMap[64];
shared int emissiveCount = 0;

uniform vec3 cameraPosition;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

void main() {
	int mat = int(gl_WorkGroupID.x);
	if (getMaterialAvailability(mat)) {
		int responsibleSize = (1<<(VOXEL_DETAIL_AMOUNT-1)) / int(gl_WorkGroupSize.x);
		if (responsibleSize == 0) responsibleSize = 1;
		ivec3 baseCoord = ivec3(gl_LocalInvocationID) * responsibleSize;
		int baseIndex = getBaseIndex(mat);
		vec3 emissiveColor = vec3(0);
		int thisEmissiveCount = 0;
		vec3 subCoord = vec3(0);
		for (int x = 0; x < responsibleSize; x++) {
			for (int y = 0; y < responsibleSize; y++) {
				for (int z = 0; z < responsibleSize; z++) {
					voxel_t thisVoxel = readGeometry(baseIndex, baseCoord + ivec3(x, y, z));
					if (thisVoxel.emissive) {
						emissiveColor += thisVoxel.color.rgb;
						subCoord += vec3(x, y, z) + 0.5001;
						thisEmissiveCount++;
					}
				}
			}
		}
		#if VOXEL_DETAIL_AMOUNT == 1
			if (thisEmissiveCount > 0) {
				setEmissiveCount(baseIndex, 1);
				storeEmissive(baseIndex, 0, ivec3(4, 4, 4));
			} else {
				setEmissiveCount(baseIndex, 0);
			}
		#else
			if (thisEmissiveCount > 0) {
				vec3 blockRelCoord = (baseCoord + subCoord / thisEmissiveCount) * (8.0 / (1<<(VOXEL_DETAIL_AMOUNT-1)));
				int sortVal = 0;
				float maxRelDist = 0;
				for (int k = 0; k < 3; k++) {
					float thisRelDist = abs(blockRelCoord[k] - 0.5);
					if (thisRelDist > maxRelDist) {
						sortVal = (k+1) * (2 * int(blockRelCoord[k] > 0) - 1);
						maxRelDist = thisRelDist;
					}
				}
				int index = atomicAdd(emissiveCount, 1);
				emissiveParts[index] = ivec4(blockRelCoord, sortVal);
			}
			int index = int(gl_LocalInvocationID.x) + int(gl_LocalInvocationID.y * gl_WorkGroupSize.x) + int(gl_LocalInvocationID.z * gl_WorkGroupSize.y * gl_WorkGroupSize.x);
			ivec4 currentVal = emissiveParts[index];
			sortMap[index] = 0;
			barrier();
			memoryBarrierShared();
			if (index < emissiveCount) {
				int sortedIndex = 0;
				for (int k = 0; k < emissiveCount; k++) {
					if (currentVal.w < emissiveParts[k].w) {
						sortedIndex++;
					}
				}
				int offset = atomicAdd(sortMap[sortedIndex], 1);
				emissiveParts[sortedIndex + offset] = currentVal;
			}
			if (index == 0) {
				setEmissiveCount(baseIndex, emissiveCount);
			}
			barrier();
			memoryBarrierShared();
			storeEmissive(baseIndex, index, index < emissiveCount ? emissiveParts[index].xyz : ivec3(-1));
		#endif
	}
}

#endif
//////2nd Compute Shader//////2nd Compute Shader//////2nd Compute Shader//////
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



#endif
