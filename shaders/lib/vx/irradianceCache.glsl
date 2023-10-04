uniform sampler3D irradianceCache;
#ifdef WRITE_TO_SSBOS
	layout(rgb16f) uniform image3D irradianceCacheI;
#endif

vec3 readIrradianceCache(vec3 vxPos, vec3 normal) {
	vec3 color = vec3(0);
	vxPos += 0.5 * voxelVolumeSize;
	for (int k = 0; k < 3; k++) {
		color += texture(irradianceCache, vxPos + vec3(0, voxelVolumeSize * (k + 3 * float(normal[k] > 0)), 0)) * abs(normal[k]);
	}
	return color;
}

#ifdef WRITE_TO_SSBOS
void writeIrradianceCache(ivec3 coord, int index, vec3 color) {
	imageStore(irradianceCacheI, coord + ivec3(0, voxelVolumeSize.y * index, 0, color));
}
#endif