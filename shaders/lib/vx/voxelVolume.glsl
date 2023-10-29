// only to be included by SSBOs.glsl
layout(r32i) uniform WRITE_TO_SSBOS iimage3D voxelVolumeI;
struct voxel_t {
	vec4 color;
	bool emissive;
	bool glColored;
};

int readBlockVolume(ivec3 coords) {
	int mat = imageLoad(voxelVolumeI, coords).r;
	return mat < MATERIALCOUNT ? mat : 0;
}

int readBlockVolume(vec3 pos) {
	return readBlockVolume(vxPosToVxCoords(pos));
}

int readGlColor(ivec3 coords) {
	return imageLoad(voxelVolumeI, coords + ivec3(0, voxelVolumeSize.y, 0)).r & 0xffffff;
}
int readLightLevel(ivec3 coords) {
	return (imageLoad(voxelVolumeI, coords + ivec3(0, voxelVolumeSize.y, 0)).r & 0x7f000000) >> 24;
}

int readEntityOccupancy(ivec3 coords) {
	coords = ivec3(2, 1, 2) * coords - ivec3(voxelVolumeSize.xz, 0).xzy / 2;
	if (any(lessThan(coords, ivec3(0))) || any(greaterThanEqual(coords, voxelVolumeSize))) {
		return 0;
	}
	return imageLoad(voxelVolumeI, coords + ivec3(0, 3 * voxelVolumeSize.y, 0)).r & 0xffff;
}

vec3[2] readEntityColor(ivec3 coords) {
	coords = ivec3(2, 1, 2) * coords - ivec3(voxelVolumeSize.xz, 0).xzy / 2;
	if (any(lessThan(coords, ivec3(0))) || any(greaterThanEqual(coords, voxelVolumeSize))) {
		return vec3[2](vec3(0), vec3(0));
	}
	ivec3 imageCoords = coords + ivec3(0, 3 * voxelVolumeSize.y, 0);
	ivec3 packedColor = ivec3(
		imageLoad(voxelVolumeI, imageCoords + ivec3(0, 0, 1)).r,
		imageLoad(voxelVolumeI, imageCoords + ivec3(1, 0, 0)).r,
		imageLoad(voxelVolumeI, imageCoords + ivec3(1, 0, 1)).r
	);
	return vec3[2](
		vec3(packedColor & 0x7fff),
		vec3(packedColor >> 15)
	);
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
	voxelData.glColored = bool(rawData >> 31 & 1u);
	voxelData.emissive  = bool(rawData >> 30 & 1u);
	voxelData.color     = vec4(
		(rawData      ) & 127u,
		(rawData >>  7) & 127u,
		(rawData >> 14) & 127u,
		(rawData >> 21) & 127u
	) * (1.0 / 127.0);
	return voxelData;
}

bool getMaterialAvailability(int mat) {
	for (int k = 0; k < 7; k++) {
		if (blockIdMap[MATERIALCOUNT + 7 * mat + k] != 0) {
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

int readEmissiveCount(int baseIndex) {
	return int(geometryData[baseIndex + modelMemorySize + (maxEmissiveVoxels + 2) / 3] & uint(0x3f));
}

#ifndef READONLY
	bool claimMaterial(int mat, int side, ivec3 coords) {
		int coordsHash = 1 + coords.x + voxelVolumeSize.x * coords.y + voxelVolumeSize.x * voxelVolumeSize.y * coords.z;
		int prevHash = atomicCompSwap(blockIdMap[MATERIALCOUNT + 7 * mat + side], 0, coordsHash);
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
			int writeIndex = baseIndex + modelMemorySize + (maxEmissiveVoxels + 2) / 3;
			atomicAnd(geometryData[writeIndex], uint(0xffffffff) ^ uint(0x3f));
			atomicOr(geometryData[writeIndex], uint(count));
		}
	}

	void storeEmissive(int baseIndex, int localIndex, ivec3 lightData) {
		uint clearData = uint(0xffffffff) ^ (uint(0x3ff) << (10*(localIndex % 3)));
		int writeIndex = baseIndex + modelMemorySize + localIndex / 3;
		atomicAnd(geometryData[writeIndex], clearData);
		if (lightData != ivec3(-1)) {
			uint data = uint(lightData.x)
			          + uint(lightData.y << 3)
			          + uint(lightData.z << 6)
			          + uint(1<<9)
			;
			data <<= 10 * (localIndex%3);
			atomicOr(geometryData[writeIndex], data);
		}
	}
#endif
