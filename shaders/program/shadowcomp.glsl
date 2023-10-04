#include "/lib/common.glsl"
//////1st Compute Shader//////1st Compute Shader//////1st Compute Shader//////
#ifdef CSH
#if VOXEL_DETAIL_AMOUNT <= 5
	const ivec3 workGroups = ivec3(16384, 1, 1);
#else
	const ivec3 workGroups = ivec3(15000, 1, 1);
#endif
#if VOXEL_DETAIL_AMOUNT == 1
	layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
#elif VOXEL_DETAIL_AMOUNT == 2
	layout(local_size_x = 2, local_size_y = 2, local_size_z = 2) in;
#else
	layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
#endif
shared ivec4 emissiveParts[64];
shared int sortMap[64];
shared int emissiveCount;
shared uvec4 totalEmissiveColor;
uniform vec3 cameraPosition;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

float getSaturation(vec3 color) {
	return 1 - min(min(color.r, color.g), color.b) / max(max(color.r, color.g), max(color.b, 0.0001));
}

void main() {
	if (gl_LocalInvocationID == uvec3(0)) {
		emissiveCount = 0;
		totalEmissiveColor = uvec4(0);
	}
	barrier();
	memoryBarrierShared();
	int mat = int(gl_WorkGroupID.x);
	bool matIsAvailable = getMaterialAvailability(mat);
	int index = int(gl_LocalInvocationID.x + gl_LocalInvocationID.y * gl_WorkGroupSize.x + gl_LocalInvocationID.z * gl_WorkGroupSize.y * gl_WorkGroupSize.x);
	ivec4 currentVal;
	int baseIndex = getBaseIndex(mat);
	int responsibleSize = 1;
	ivec3 baseCoord = ivec3(0);
	if (matIsAvailable) {
		responsibleSize = (1<<(VOXEL_DETAIL_AMOUNT-1)) / int(gl_WorkGroupSize.x);
		if (responsibleSize == 0) responsibleSize = 1;
		baseCoord = ivec3(gl_LocalInvocationID) * responsibleSize;
		for (int x = 0; x < responsibleSize; x++) {
			for (int y = 0; y < responsibleSize; y++) {
				for (int z = 0; z < responsibleSize; z++) {
					voxel_t thisVoxel = readGeometry(baseIndex, baseCoord + ivec3(x, y, z));
					if (thisVoxel.emissive) {
						for (int i = 0; i < 3; i++) {
							atomicAdd(totalEmissiveColor[i], uint(thisVoxel.color[i] * thisVoxel.color[i] * 255 + 0.5));
						}
						atomicAdd(totalEmissiveColor.w, 255);
					}
				}
			}
		}
	}
	barrier();
	memoryBarrierShared();
	vec3 meanEmissiveColor = sqrt(vec3(totalEmissiveColor.rgb) / max(totalEmissiveColor.a, 1));
	#if RP_MODE <= 1
		barrier();
		memoryBarrierShared();
		if (gl_LocalInvocationID == uvec3(0)) {
			totalEmissiveColor = uvec4(0);
		}
		barrier();
		memoryBarrierShared();
		if (matIsAvailable) {
			float meanEmissiveLuminance = max(max(meanEmissiveColor.r, meanEmissiveColor.g), meanEmissiveColor.b);
			float meanEmissiveSaturation = getSaturation(meanEmissiveColor) * meanEmissiveLuminance;
			float threshold = meanEmissiveLuminance + meanEmissiveSaturation;
			for (int x = 0; x < responsibleSize; x++) {
				for (int y = 0; y < responsibleSize; y++) {
					for (int z = 0; z < responsibleSize; z++) {
						voxel_t thisVoxel = readGeometry(baseIndex, baseCoord + ivec3(x, y, z));
						if (thisVoxel.emissive) {
							float thisLuminance = max(max(thisVoxel.color.r, thisVoxel.color.g), thisVoxel.color.b);
							float thisSaturation = getSaturation(thisVoxel.color.rgb) * thisLuminance;
							if (thisLuminance + thisSaturation > 1.2 * threshold || thisLuminance > min(0.3 * meanEmissiveLuminance + 0.7, 0.8) || thisSaturation > min(0.3 * meanEmissiveSaturation + 0.7, 0.8)) {
								for (int i = 0; i < 3; i++) {
									atomicAdd(totalEmissiveColor[i], uint(thisVoxel.color[i] * thisVoxel.color[i] * 255 + 0.5));
								}
								atomicAdd(totalEmissiveColor.w, 255);
							} else {
								thisVoxel.emissive = false;
								writeGeometry(baseIndex, (baseCoord + ivec3(x, y, z)) * (1.0 / (1<<(VOXEL_DETAIL_AMOUNT - 1))), thisVoxel);
							}
						}
					}
				}
			}
		}
		barrier();
		memoryBarrierShared();
		memoryBarrierBuffer();
	#endif
	meanEmissiveColor = sqrt(vec3(totalEmissiveColor.rgb) / max(totalEmissiveColor.a, 1));
	if (matIsAvailable) {
		float meanEmissiveColorMax = max(max(meanEmissiveColor.r, meanEmissiveColor.g), meanEmissiveColor.b);
		float meanEmissiveColorMin = min(min(meanEmissiveColor.r, meanEmissiveColor.g), meanEmissiveColor.b);
		vec3 saturatedMeanEmissiveColor = meanEmissiveColorMax + meanEmissiveColorMax / max(meanEmissiveColorMax - meanEmissiveColorMin, 0.001) * (meanEmissiveColor - meanEmissiveColorMax);
		saturatedMeanEmissiveColor = mix(meanEmissiveColor, saturatedMeanEmissiveColor, LIGHT_COLOR_SATURATION);
		for (int x = 0; x < responsibleSize; x++) {
			for (int y = 0; y < responsibleSize; y++) {
				for (int z = 0; z < responsibleSize; z++) {
					voxel_t thisVoxel = readGeometry(baseIndex, baseCoord + ivec3(x, y, z));
					if (thisVoxel.emissive) {
						thisVoxel.color.rgb = mix(thisVoxel.color.rgb, saturatedMeanEmissiveColor, 0.5);
						thisVoxel.color.a = 1.0;
						writeGeometry(baseIndex, (baseCoord + ivec3(x, y, z)) * (1.0 / (1<<(VOXEL_DETAIL_AMOUNT - 1))), thisVoxel);
					}
				}
			}
		}
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
		#endif
	}
	#if VOXEL_DETAIL_AMOUNT > 1
		if (index == 0) {
			setEmissiveCount(baseIndex, emissiveCount);
		}
		barrier();
		memoryBarrierShared();
		storeEmissive(baseIndex, index, index < emissiveCount ? emissiveParts[index].xyz : ivec3(-1));
	#endif
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

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

#include "/lib/util/random.glsl"

void main() {

}
#endif
