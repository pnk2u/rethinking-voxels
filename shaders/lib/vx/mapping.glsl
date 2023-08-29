#ifdef DECLARE_CAMPOS
	uniform vec3 cameraPosition;
	#ifdef PREV_FRAME_MAPPING
		uniform vec3 previousCameraPosition;
	#endif
#endif

vec3 playerToVx(vec3 player) {
	return player + fract(cameraPosition);
}
vec3 worldToVx(vec3 world) {
	return world + floor(cameraPosition);
}

#ifdef PREV_FRAME_MAPPING
	vec3 playerToPreviousVx(vec3 player) {
		return player + (cameraPosition - floor(previousCameraPosition));
	}
	vec3 worldToPreviousVx(vec3 world) {
		return world + floor(previousCameraPosition);
	}
#endif

ivec3 vxPosToVxCoords(vec3 vxPos, int lod) {
	vxPos *= 1<<lod;
	vxPos += 0.5 * voxelVolumeSize;
	if (all(greaterThan(vxPos, vec3(0))) && all(lessThan(vxPos, voxelVolumeSize))) {
		return ivec3(vxPos);
	}
	return ivec3(-1);
}

int maxAvailableLod(vec3 vxPos) {
	vxPos = abs(vxPos / voxelVolumeSize);
	for (int k = voxelDetailAmount - 1; k >= 0; k--) {
		if (all(lessThan(vxPos * (1<<k), vec3(0.5)))) {
			return k;
		}
	}
	return -1;
}
