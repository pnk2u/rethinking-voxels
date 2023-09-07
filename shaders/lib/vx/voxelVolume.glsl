// only to be included by SSBOs.glsl
layout(r32ui) uniform WRITE_TO_SSBOS uimage3D voxelVolumeI;
struct voxel_t {
	vec4 color;
	bool emissive;
	bool full;
};

uint readBlockVolume(ivec3 coords) {
	return imageLoad(voxelVolumeI, coords).r;
}

voxel_t readGeometry(int mat, vec3 pos, int lod) {
	int lodSubdivisions = 1<<lod;
	ivec3 coord = ivec3(pos * lodSubdivisions);
	int index = modelMemorySize * mat
	          + coord.x * lodSubdivisions * lodSubdivisions
	          + coord.y * lodSubdivisions
	          + coord.z;
	for (int i = 0; i < lod; i++) {
		index += (1<<i) * (1<<i) * (1<<i);
	}
	uint rawData = geometryData[index];
	voxel_t voxelData;
	voxelData.full     = ((rawData >> 31) % 2 == 0);
	voxelData.emissive = ((rawData >> 30) % 2 != 0);
	voxelData.color    = vec4(
		(rawData      ) % 128,
		(rawData >>  7) % 128,
		(rawData >> 14) % 128,
		(rawData >> 21) % 128
	) / 128;
	return voxelData;
}

bool getMaterialAvailability(int mat) {
	return (materialMap[16384 + 7 * mat] != 0);
}

#ifndef READONLY
	bool claimMaterial(int mat, int side, ivec3 coords) {
		int coordsHash = 1 + coords.x + voxelVolumeSize.x * coords.y + voxelVolumeSize.x * voxelVolumeSize.y * coords.z;
		int prevHash = atomicCompSwap(materialMap[16384 + 7 * mat + side], 0, coordsHash);
		if (prevHash == 0 || prevHash == coordsHash) return true;
		return false;
	}

	void writeGeometry(int mat, vec3 pos, int lod, voxel_t data) {
		int lodSubdivisions = 1<<lod;
		ivec3 coord = ivec3(pos * lodSubdivisions);
		int index = modelMemorySize * mat + coord.x * lodSubdivisions * lodSubdivisions + coord.y * lodSubdivisions + coord.z;
		for (int i = 0; i < lod; i++) {
			index += (1<<i) * (1<<i) * (1<<i);
		}
		uint rawData
			=              int(data.color.r * 127.9)
			+        128 * int(data.color.g * 127.9)
			+      16384 * int(data.color.b * 127.9)
			+    2097152 * int(data.color.a * 127.9)
			+ 1073741824 * int(data.emissive)
			+ 2147483648 * int(!data.full);
		geometryData[index] = rawData;
	}
#endif
