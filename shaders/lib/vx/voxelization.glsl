for (int _lkakmdffonef = 0; _lkakmdffonef < 1; _lkakmdffonef++) {
    vec3 vxPos[3];
    vec3 bounds[2] = vec3[2](vec3(1e10), vec3(-1e10));
    for (int i = 0; i < 3; i++) {
        vxPos[i] = playerToVx(positionV[i].xyz);
        bounds[0] = min(bounds[0], vxPos[i]);
        bounds[1] = max(bounds[1], vxPos[i]);
    }

    if (!isInRange(bounds[0]) || !isInRange(bounds[1])) break;
    vec3 normal = cross(vxPos[1] - vxPos[0], vxPos[2] - vxPos[0]);
    float area = max(length(normal), 1e-10);
    normal /= area;

    vec3 center = 0.5 * (bounds[1] + bounds[0]);
    vec3 size = bounds[1] - bounds[0];
    for (int i = 0; i < 3; i++) {
        vxPos[i] = mix(vxPos[i], center, 0.01) - 0.03 * normal;
    }

    // center cross-model blocks
    if (abs(normal.y) < 0.1 && abs(abs(normal.x) - abs(normal.z)) < 0.1) {
        for (int i = 0; i < 3; i++) {
            vxPos[i].xz += 0.5 - fract(center.xz);
        }
    }

    ivec3 correspondingBlock = ivec3(vxPos[0] + midBlock[0] + vec3(1024)) - ivec3(1024);

    if (length(max(max(abs(fract(vxPos[0] + midBlock[0]) - 0.5),
        abs(fract(vxPos[1] + midBlock[1]) - 0.5)),
        abs(fract(vxPos[2] + midBlock[2]) - 0.5))) > 0.1
    ) {
        correspondingBlock = ivec3(center - 0.05 * normal + vec3(1024)) - ivec3(1024);
        if (correspondingBlock != ivec3(center - 0.01 * normal + vec3(1024)) - ivec3(1024) && area < 0.1) {
            break;
        }
    }

    ivec3 blockCoords = vxPosToVxCoords(correspondingBlock + 0.5);

    int processedMat = getProcessedBlockId(matV[0]);

    if (currentRenderedItemId * HELD_LIGHTING_MODE == 0 && ignoreMat(processedMat)) {
        break;
    }

    bool matIsEmissive = isEmissive(processedMat);
    int lightLevel = matIsEmissive ? getLightLevel(processedMat) : 0;
    if (lightLevel == 0) lightLevel = int(lmCoordV[0].x * lmCoordV[0].x * 18) + 7;
    if (matV[0] == 0 || matV[0] > MATERIALCOUNT) {// unknown blocks and entities
        int processedItemId = 0;
        bool itemIsEmissive = false;
        if (currentRenderedItemId > 0) {
            processedItemId = getProcessedBlockId(currentRenderedItemId);
            matIsEmissive = isEmissive(processedItemId);
            if (matIsEmissive) {
                lightLevel = getLightLevel(processedItemId);
                #if HELD_LIGHTING_MODE < 2
                    lightLevel = lightLevel / 3;
                #endif
            }
        }
        if (processedMat == 50016) {
            #if HELD_LIGHTING_MODE > 0
                if (lightLevel == 0) break;
                #if HELD_LIGHTING_MODE == 1
                    lightLevel = lightLevel / 3;
                #endif
            #else
                break;
            #endif
        }
        if (!matIsEmissive && (processedItemId > 0 || processedMat > MATERIALCOUNT)) {
            break;
        }
        float shortestEdge = min(min(
            length(vxPos[1] - vxPos[0]),
            length(vxPos[2] - vxPos[1])),
            length(vxPos[0] - vxPos[2])
        );
        if (shortestEdge < 0.1 || area < 0.05) {
            break;
        }
        vec4 color = textureLod(tex, 0.5 * (max(max(texCoordV[0], texCoordV[1]), texCoordV[2]) + min(min(texCoordV[0], texCoordV[1]), texCoordV[2])), 2);
        if (color.a < 0.4) {
            break;
        }
        if (matIsEmissive) {
            vec3 HCLightCol = getLightCol(processedItemId > 0 ? processedItemId : processedMat);
            color.rgb = mix(color.rgb, HCLightCol, min(1, length(HCLightCol)));
        }
        vec3 faceCenterPosM = center - 0.3 * sqrt(area) * normal;
        ivec3 vxCoordM0 = vxPosToVxCoords(faceCenterPosM);
        ivec3 vxCoordM = ivec3(2, 1, 2) * vxCoordM0 - ivec3(voxelVolumeSize.xz, 0).xzy / 2;
        if (any(lessThan(vxCoordM, ivec3(0))) || any(greaterThanEqual(vxCoordM, voxelVolumeSize))) {
            continue;
        }
        ivec3 subBlockCoord = ivec3(0.2 * fract(faceCenterPosM) + 0.9);
        int index = subBlockCoord.x + 2 * subBlockCoord.y + 4 * subBlockCoord.z;
        ivec3 imageCoord = vxCoordM + ivec3(0, 3 * voxelVolumeSize.y, 0);
        imageAtomicOr(
            voxelVolumeI,
            imageCoord,
            (1<<index) + (int(matIsEmissive) << (index + 8))
        );
        imageAtomicAdd(voxelVolumeI, imageCoord, 1<<16);
        for (int k = 1; k < 4; k++) {
            imageAtomicAdd(
                voxelVolumeI,
                imageCoord + ivec3(k/2, 0, k%2),
                int(63 * color[k-1] + 0.5)
            );
        }
        if (matIsEmissive) {
            imageAtomicAnd(voxelVolumeI, vxCoordM0 + ivec3(0, voxelVolumeSize.y, 0), 0x80ffffff);
            imageAtomicOr(voxelVolumeI, vxCoordM0 + ivec3(0, voxelVolumeSize.y, 0), lightLevel << 24);
        }
        break;
    }

    uint prevMat = imageAtomicCompSwap(voxelVolumeI, blockCoords, 0, matV[0]);
    if (prevMat != 0 && prevMat != uint(matV[0])) break;

    vec4 meanGlColor = 0.333 * (glColorV[0] + glColorV[1] + glColorV[2]);
    bool hasGlColor = (length(meanGlColor.rgb - vec3(1)) > 0.1);
    meanGlColor.rgb = sqrt(meanGlColor.rgb);
    if (hasGlColor) {
        int packedGlColor =  int(255.5 * meanGlColor.r)
                          + (int(255.5 * meanGlColor.g) << 8)
                          + (int(255.5 * meanGlColor.b) << 16);
        imageAtomicAnd(voxelVolumeI, blockCoords + ivec3(0, voxelVolumeSize.y, 0), 0xff000000);
        imageAtomicOr(voxelVolumeI, blockCoords + ivec3(0, voxelVolumeSize.y, 0), packedGlColor);
    }

    imageAtomicAnd(voxelVolumeI, blockCoords + ivec3(0, voxelVolumeSize.y, 0), 0x80ffffff);
    imageAtomicOr(voxelVolumeI, blockCoords + ivec3(0, voxelVolumeSize.y, 0), lightLevel << 24);

    vec3 blockRelPos[3];
    float sizeHeuristic = sqrt(area);
    int mostPerpendicularAxis = 0;
    for (int i = 0; i < 3; i++) {
        // figure out position relative to the block origin
        blockRelPos[i] = vxPos[i] - correspondingBlock;
        // figure out where the face is facing
        if (abs(normal[i]) > abs(normal[mostPerpendicularAxis])) {
            mostPerpendicularAxis = i;
        }
    }

    int side = 0;
    for (int k = 0; k < 6; k++) {
        vec3 dir = mat3(k/3 * 2 - 1)[k%3];
        if (dot(dir, normal) > 0.9 && dot(dir, center - correspondingBlock - 0.5) > 0.45) {
            side = k+1;
        }
    }
    if (!claimMaterial(matV[0], side, blockCoords)) break;

    mat3x2 blockRelToProjected = mat3x2(0);
    for (int i = 0; i < 2; i++) {
        blockRelToProjected[(mostPerpendicularAxis + i + 1) % 3][i] = 1;
    }

    vec2[3] projectedPos;
    for (int i = 0; i < 3; i++) {
        projectedPos[i] = blockRelToProjected * blockRelPos[i];
    }
    mat3 localToProjected =
        mat3(vec3(projectedPos[1] - projectedPos[0], 0),
            vec3(projectedPos[2] - projectedPos[0], 0),
            vec3(projectedPos[0], 1));
    mat3 projectedToLocal = inverse(localToProjected);
    vec3 depthFromLocal =
        vec3(blockRelPos[1][mostPerpendicularAxis] -
            blockRelPos[0][mostPerpendicularAxis],
            blockRelPos[2][mostPerpendicularAxis] -
            blockRelPos[0][mostPerpendicularAxis],
            blockRelPos[0][mostPerpendicularAxis]);

    mat3 projectedToBlockRel = mat3(0);
    for (int i = 0; i < 2; i++) {
        projectedToBlockRel[(mostPerpendicularAxis + i + 1) % 3][i] = 1;
    }
    projectedToBlockRel[mostPerpendicularAxis] = depthFromLocal * projectedToLocal;
    projectedToBlockRel = transpose(projectedToBlockRel);
    mat3 textureMatrix = mat3(vec3(texCoordV[1] - texCoordV[0], 0),
                            vec3(texCoordV[2] - texCoordV[0], 0),
                            vec3(texCoordV[0], 1));

    ivec3 boundCoords[2] = ivec3[2](
        ivec3(max(vec3(0), (bounds[0] - correspondingBlock)) * (1<<(VOXEL_DETAIL_AMOUNT-1))),
        ivec3(min(vec3(0.9999), (bounds[1] - correspondingBlock)) * (1<<(VOXEL_DETAIL_AMOUNT-1)))
    );

    ivec2 projectionBoundCoords[2];
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            projectionBoundCoords[i][j] =
                boundCoords[i]
                            [(mostPerpendicularAxis + j + 1) % 3];
        }
    }
    bool smallFace = any(lessThan(projectionBoundCoords[1] - projectionBoundCoords[0], ivec2(2)));
    float offTriMargin = 0.1 / dot(vec2(1), projectionBoundCoords[1] - projectionBoundCoords[0] + 0.5);
    int baseIndex = getBaseIndex(matV[0]);
    for (int x = projectionBoundCoords[0].x;
            x <= projectionBoundCoords[1].x; x++) {
        for (int y = projectionBoundCoords[0].y;
                y <= projectionBoundCoords[1].y; y++) {
            vec3 thisProjectedPos = vec3(x + 0.5, y + 0.5, 1) / vec2(1 << (VOXEL_DETAIL_AMOUNT-1), 1).xxy;
            vec3 localPos = projectedToLocal * thisProjectedPos;
            if (!smallFace && (localPos.x < -offTriMargin || localPos.y < -offTriMargin ||
                localPos.x + localPos.y > 1.0 + offTriMargin)) {
                continue;
            }
            localPos.xy = clamp(localPos.xy, 0.01, 0.99);
            if (localPos.x + localPos.y > 1.0 / 1.001) {
                localPos.xy /= (localPos.x + localPos.y) * 1.001;
            }
            vec2 thisTexCoord = (textureMatrix * localPos).xy;
            vec4 color = textureLod(tex, thisTexCoord, 0);
            vec4 s = textureLod(specular, thisTexCoord, 0);
            if (s.a > 0.999) s.a = 0;
            if (s.a > 0.1)  {
                color.rgb *= s.a;
            }
            vec3 thisPos = projectedToBlockRel * thisProjectedPos;
            if (color.a < 0.1 || thisPos != clamp(thisPos, 0, 1) || badPixel(color, meanGlColor, processedMat)) {
                continue;
            }

            voxel_t voxelData;
            voxelData.color = vec4(color.rgb, color.a);
            voxelData.glColored = hasGlColor;
            #if RP_MODE <= 1
                voxelData.emissive = matIsEmissive;
            #else
                voxelData.emissive = s.a > 0.1;
            #endif
            writeGeometry(baseIndex, thisPos, voxelData);
        }
    }
}
