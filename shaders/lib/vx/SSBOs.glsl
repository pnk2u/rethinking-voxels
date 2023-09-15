#ifndef SSBO
	#define SSBO
	#ifndef WRITE_TO_SSBOS
		#define READONLY
		#define WRITE_TO_SSBOS readonly
	#endif
	// constants
	#ifndef MATERIALMAP_ONLY
		#if VX_VOL_SIZE == 0
			const ivec3 voxelVolumeSize = ivec3(96, 64, 96);
		#elif VX_VOL_SIZE == 1
			const ivec3 voxelVolumeSize = ivec3(128, 96, 128);
		#elif VX_VOL_SIZE == 2
			const ivec3 voxelVolumeSize = ivec3(256, 128, 256);
		#elif VX_VOL_SIZE == 3
			const ivec3 voxelVolumeSize = ivec3(512, 128, 512);
		#endif
		const int modelMemorySize = (1<<(VOXEL_DETAIL_AMOUNT-1)) * (1<<(VOXEL_DETAIL_AMOUNT-1)) * (1<<(VOXEL_DETAIL_AMOUNT-1));
	#endif

	layout(std430, binding=0) WRITE_TO_SSBOS buffer blockidmap {
		mat4 gbufferPreviousModelViewInverse;
		mat4 gbufferPreviousProjectionInverse;
		int blockIdMap[];
	};

	#ifndef MATERIALMAP_ONLY
		layout(std430, binding=1) WRITE_TO_SSBOS buffer geometrydata {
			uint geometryData[];
		};
		// voxelisation-related mapping functions
		#include "/lib/vx/mapping.glsl"

		// voxel volume
		#include "/lib/vx/voxelVolume.glsl"
	#endif
#endif
