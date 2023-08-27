// only to be included by SSBOs.glsl
layout(r32ui) uniform WRITE_TO_SSBOS uimage3D voxelVolumeI;
struct voxel_t {
	vec4 color;
	bool emissive;
	bool full;
};

voxel_t readVoxelVolume(ivec3 coords, int lod) {
	uint rawData = imageLoad(
		voxelVolumeI,
		coords + ivec3(0, voxelVolumeSize.y * lod, 0)
	).r;
	voxel_t voxelData;
	voxelData.full     = ((rawData >> 31) % 2 == 0);
	voxelData.emissive = ((rawData >> 30) % 2 != 0);
	voxelData.color    = vec4(
		(rawData      ) % 128,
		(rawData >>  7) % 128,
		(rawData >> 14) % 128,
		(rawData >> 21) % 128
	);
	return voxelData;
}

#ifndef READONLY
	void writeVoxelVolume(ivec3 coords, int lod, voxel_t voxelData) {
		uint rawData
			=              int(voxelData.color.r * 127.9)
			+        128 * int(voxelData.color.g * 127.9)
			+      16384 * int(voxelData.color.b * 127.9)
			+    2097152 * int(voxelData.color.a * 127.9)
			+ 1073741824 * int(voxelData.emissive)
			+ 2147483648 * int(!voxelData.full);
		imageStore(
			voxelVolumeI,
			coords + ivec3(0, voxelVolumeSize.y * lod, 0),
			uvec4(rawData)
		);
	}
#endif
