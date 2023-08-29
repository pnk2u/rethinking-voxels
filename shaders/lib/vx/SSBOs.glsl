#ifndef SSBO
	#define SSBO
	#ifndef WRITE_TO_SSBOS
		#define READONLY
		#define WRITE_TO_SSBOS readonly
	#endif

	// constants
	#if VX_VOL_SIZE == 0
		const ivec3 voxelVolumeSize = ivec3(96, 64, 96);
	#elif VX_VOL_SIZE == 1
		const ivec3 voxelVolumeSize = ivec3(128, 96, 128);
	#elif VX_VOL_SIZE == 2
		const ivec3 voxelVolumeSize = ivec3(256, 128, 256);
	#elif VX_VOL_SIZE == 3
		const ivec3 voxelVolumeSize = ivec3(512, 128, 512);
	#endif
	// voxelisation-related mapping functions
	#include "/lib/vx/mapping.glsl"

	// voxel volume
	#include "/lib/vx/voxelVolume.glsl"
#endif
