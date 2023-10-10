#ifdef WRITE_TO_SSBOS
	layout(rgba16f) uniform image3D irradianceCacheI;
#else
	uniform sampler3D irradianceCache;
#endif

#ifndef WRITE_TO_SSBOS
	vec3 readIrradianceCache(vec3 vxPos, vec3 normal) {
		vxPos = ((vxPos + 0.5 * normal) / voxelVolumeSize + 0.5) * vec3(1.0, 0.5, 1.0);
		vec4  color = textureLod(irradianceCache, vxPos, 0);
		return color.rgb / color.a;
	}

	vec3 readVolumetricBlocklight(vec3 vxPos) {
		vxPos = (vxPos / voxelVolumeSize + vec3(0.5, 1.5, 0.5)) * vec3(1.0, 0.5, 1.0);
		vec4 color = textureLod(irradianceCache, vxPos, 0);
		return color.rgb / color.a;
	}
#endif