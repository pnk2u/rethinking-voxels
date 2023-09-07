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

bool isInRange(vec3 vxPos) {
	return all(greaterThan(vxPos, -0.5 * voxelVolumeSize)) && all(lessThan(vxPos, 0.5 * voxelVolumeSize));
}

ivec3 vxPosToVxCoords(vec3 vxPos) {
	if (isInRange(vxPos)) {
		vxPos += 0.5 * voxelVolumeSize;
		return ivec3(vxPos);
	}
	return ivec3(-1);
}
