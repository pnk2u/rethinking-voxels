#ifndef SSBO
	#define SSBO
	#ifndef WRITE_TO_SSBOS
		#define READONLY
		#define WRITE_TO_SSBOS readonly
	#endif

	// constants
	const ivec3 voxelVolumeSize = ivec3(256, 128, 256);
	const int voxelDetailAmount = 6;

	// voxelisation-related mapping functions
	#include "/lib/vx/mapping.glsl"

	// voxel volume
	#include "/lib/vx/voxelVolume.glsl"
#endif