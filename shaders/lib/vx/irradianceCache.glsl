#ifndef READONLY
    layout(rgba16f) uniform image3D irradianceCacheI;
#else
    uniform sampler3D irradianceCache;

    vec3 readIrradianceCache(vec3 vxPos, vec3 normal) {
        if (!isInRange(vxPos)) return vec3(0);
        vxPos = ((vxPos + 0.5 * normal) / voxelVolumeSize + 0.5) * vec3(1.0, 0.5, 1.0);
        vec4 color = textureLod(irradianceCache, vxPos, 0);
        return color.rgb / max(color.a, 0.0001);
    }

    vec3 readSurfaceVoxelBlocklight(vec3 vxPos, vec3 normal) {
        if (!isInRange(vxPos)) return vec3(0);
        vxPos = ((vxPos + 0.5 * normal) / voxelVolumeSize + vec3(0.5, 1.5, 0.5)) * vec3(1.0, 0.5, 1.0);
        vec4 color = textureLod(irradianceCache, vxPos, 0);
        return color.rgb / max(color.a, 0.0001);
    }

    vec3 readVolumetricBlocklight(vec3 vxPos) {
        if (!isInRange(vxPos)) return vec3(0);
        vxPos = (vxPos / voxelVolumeSize + vec3(0.5, 1.5, 0.5)) * vec3(1.0, 0.5, 1.0);
        vec4 color = textureLod(irradianceCache, vxPos, 0);
        return color.rgb / max(color.a, 0.0001);
    }
#endif