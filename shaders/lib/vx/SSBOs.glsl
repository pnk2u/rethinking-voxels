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
		#if VOXEL_DETAIL_AMOUNT == 1
			const int maxEmissiveVoxels = 1;
		#elif VOXEL_DETAIL_AMOUNT == 2
			const int maxEmissiveVoxels = 8;
		#else
			const int maxEmissiveVoxels = 64;
		#endif
	#endif

	layout(std430, binding=0) WRITE_TO_SSBOS buffer blockidmap {
		mat4 gbufferPreviousModelViewInverse;
		mat4 gbufferPreviousProjectionInverse;
		mat4 reprojectionMatrix;
		int blockIdMap[];
	};
	int getProcessedBlockId(int mat) {
		mat = blockIdMap[mat];
		return mat/10000*10000 + mat/4*4%2000;
	}

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
