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

int readBlockVolume(vec3 pos) {
	return readBlockVolume(vxPosToVxCoords(pos));
}

int readGlColor(ivec3 coords) {
	return imageLoad(voxelVolumeI, coords + ivec3(0, voxelVolumeSize.y, 0)).r;
}

int getBaseIndex(int mat) {
	return (modelMemorySize + (maxEmissiveVoxels + 2) / 3 + 1) * mat;
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

vec3 readEmissiveLoc(int baseIndex, int localIndex) {
	uint rawData = geometryData[baseIndex + modelMemorySize + localIndex / 3];
	int offset = 10 * (localIndex%3);
	if ((rawData & (uint(1)<<(offset + 9))) == 0) {
		return vec3(-1);
	}
	return vec3(int(rawData >> offset) & 7, int(rawData >> (offset + 3)) & 7, int(rawData >> (offset + 6)) & 7) * 0.125 + 0.0625;
}

int getEmissiveCount(int baseIndex) {
	return int(geometryData[baseIndex + modelMemorySize + (maxEmissiveVoxels + 2) / 3]) & 0x3f;
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
			+ (uint(data.glColored)       << 31)
		;
		geometryData[index] = rawData;
	}

	void setEmissiveCount(int baseIndex, int count) {
		if (count < 64) {
			atomicAnd(geometryData[baseIndex + modelMemorySize + (maxEmissiveVoxels + 2) / 3], uint(0xffffffff) ^ uint(0x3f));
			atomicOr(geometryData[baseIndex + modelMemorySize + (maxEmissiveVoxels + 2) / 3], uint(count));
		}
	}

	void setEmissiveDirectionRanges(int baseIndex, int ranges[6]) {
	}

	void storeEmissive(int baseIndex, int localIndex, ivec3 lightData) {
		uint clearData = uint(0xffffffff) ^ (uint(0x3ff) << (10*(localIndex % 3)));
		atomicAnd(geometryData[baseIndex + modelMemorySize + localIndex / 3], clearData);
		if (lightData != ivec3(-1)) {
			uint data = lightData.x
			          + (lightData.y << 3);
			          + (lightData.z << 6);
			          + (1<<9)
			;
			data <<= 10 * (localIndex%3);
			atomicOr(geometryData[baseIndex + modelMemorySize + localIndex / 3], data);
		}
	}
#endif
