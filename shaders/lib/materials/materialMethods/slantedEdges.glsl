#ifdef GBUFFERS_TERRAIN
    #ifdef ACL_VOXELIZATION
        #include "/lib/misc/voxelization.glsl"
    #endif
    void GenerateEdgeSlopes(inout vec3 normalM, vec3 playerPos) {
        normalM = mat3(gbufferModelViewInverse) * normalM;
        vec3 voxelPos = SceneToVoxel(playerPos);
        vec3 blockRelPos = fract(voxelPos - 0.1 * normalM) - 0.5;
        ivec3 edge = ivec3(greaterThan(abs(blockRelPos) - 0.5 + 0.0625 - abs(normalM), vec3(0.0))) * ivec3(sign(blockRelPos) * 1.01);
        if (max(max(abs(normalM.x), abs(normalM.y)), abs(normalM.z)) < 0.99) {
            edge = ivec3(0);
        }
        #ifdef ACL_VOXELIZATION
            if (edge != ivec3(0) && CheckInsideVoxelVolume(voxelPos)) {
                ivec3 coords = ivec3(voxelPos - 0.1 * normalM);
                uint this_mat = texelFetch(voxel_sampler, coords, 0).r;
                for (int k = 0; k < 3; k++) {
                    if (edge[k] == 0) continue;
                    ivec3 offset = edge * ivec3(equal(ivec3(k), ivec3(0, 1, 2)));
                    uint other_mat = texelFetch(voxel_sampler, coords + offset, 0).r;
                    uint above_mat = texelFetch(voxel_sampler, coords + offset + ivec3(1.01 * normalM), 0).r;
                    if (this_mat == above_mat) {
                        edge[k] *= -1;
                    } else if (this_mat == other_mat) {
                        edge[k] = 0;
                    }
                }
            }
        #endif
        normalM = mat3(gbufferModelView) * normalize(normalM + edge);
    }
#else
    void GenerateEdgeSlopes(inout vec3 normalM) {
        #ifdef GBUFFERS_ENTITIES
            ivec2 atlasSize = textureSize(tex, 0);
        #endif
        vec2 spriteSize = atlasSize * absMidCoordPos;
        vec2 edge = vec2(greaterThan(abs(signMidCoordPos * spriteSize) - max(spriteSize, vec2(2)) + 1.0, vec2(0.0))) * sign(signMidCoordPos);
        normalM = normalize(
            normalM +
            edge.x * tangent +
            edge.y * binormal
        );
    }
#endif
