// only to be included by SSBOs.glsl
layout(r32i) uniform WRITE_TO_SSBOS iimage3D voxelVolumeI;
struct voxel_t {
	vec4 color;
	bool emissive;
	bool glColored;
};

int readBlockVolume(ivec3 coords) {
	return imageLoad(voxelVolumeI, coords).r;
}

int readGlColor(ivec3 coords) {
	return imageLoad(voxelVolumeI, coords + ivec3(0, voxelVolumeSize.y, 0)).r;
}
int readBlockVolume(vec3 pos) {
	return readBlockVolume(vxPosToVxCoords(pos));
}

int getBaseIndex(int mat) {
	return modelMemorySize * mat;
}

voxel_t readGeometry(int index, ivec3 coord) {
	int lodSubdivisions = 1<<(VOXEL_DETAIL_AMOUNT-1);
	index += coord.x * lodSubdivisions * lodSubdivisions
	      +  coord.y * lodSubdivisions
	      +  coord.z;
	uint rawData = geometryData[index];
	voxel_t voxelData;
	voxelData.glColored = ((rawData >> 31) % 2 != 0);
	voxelData.emissive  = ((rawData >> 30) % 2 != 0);
	voxelData.color     = vec4(
		(rawData      ) % 128,
		(rawData >>  7) % 128,
		(rawData >> 14) % 128,
		(rawData >> 21) % 128
	) / 127.0;
	return voxelData;
}

bool getMaterialAvailability(int mat) {
	for (int k = 0; k < 7; k++) {
		if (blockIdMap[16384 + 7 * mat + k] != 0) {
			return true;
		}
	}
	return false;
}

#ifndef READONLY
	bool claimMaterial(int mat, int side, ivec3 coords) {
		int coordsHash = 1 + coords.x + voxelVolumeSize.x * coords.y + voxelVolumeSize.x * voxelVolumeSize.y * coords.z;
		int prevHash = atomicCompSwap(blockIdMap[16384 + 7 * mat + side], 0, coordsHash);
		if (prevHash == 0 || prevHash == coordsHash) return true;
		return false;
	}

	void writeGeometry(int index, vec3 pos, voxel_t data) {
		int lodSubdivisions = 1<<(VOXEL_DETAIL_AMOUNT-1);
		ivec3 coord = ivec3(pos * lodSubdivisions);
		index += coord.x * lodSubdivisions * lodSubdivisions + coord.y * lodSubdivisions + coord.z;

		uint rawData
			=  uint(data.color.r * 127.5)
			+ (uint(data.color.g * 127.5) <<  7)
			+ (uint(data.color.b * 127.5) << 14)
			+ (uint(data.color.a * 127.5) << 21)
			+ (uint(data.emissive)        << 30)
			+ (uint(data.glColored)      << 31)
		;
		geometryData[index] = rawData;
	}
#endif
