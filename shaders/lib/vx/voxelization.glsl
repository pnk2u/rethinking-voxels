do {
	vec3 vxPos[3];
	vec3 bounds[2] = vec3[2](vec3(1e10), vec3(-1e10));
	for (int i = 0; i < 3; i++) {
		vxPos[i] = playerToVx(positionV[i].xyz);
		bounds[0] = min(bounds[0], vxPos[i]);
		bounds[0] = max(bounds[1], vxPos[i]);
	}

	if (!isInRange(bounds[0]) || isInRange(bounds[1])) break;
	vec3 normal = cross(vxPos[1] - vxPos[0], vxPos[2] - vxPos[0]);
	float area = max(length(normal), 1e-10);
	normal /= area;

	vec3 center = 0.5 * (bounds[1] + bounds[0]);
	vec3 size = bounds[1] - bounds[0];
	for (int i = 0; i < 3; i++) {
		vxPos[i] = mix(vxPos[i], center, 0.01);
	}

	ivec3 correspondingBlock = ivec3(vxPos[0] + midBlock[0] + vec3(1024)) - ivec3(1024);

	if (all(lessThan(abs(midBlock[0]), vec3(0.05))) &&
		all(lessThan(abs(midBlock[1]), vec3(0.05))) &&
		all(lessThan(abs(midBlock[2]), vec3(0.05)))
	) {
		correspondingBlock = ivec3(center - 0.05 * normal + vec3(1024)) - ivec3(1024);
		if (correspondingBlock != ivec3(center - 0.01 * normal + vec3(1024)) - ivec3(1024) && area < 0.1) {
			break;
		}
	}
	ivec3 blockCoords = correspondingBlock + voxelVolumeSize / 2;
	uint prevMat = imageAtomicCompSwap(voxelVolumeI, coords, 0, uint(matV[0]));
	if (prevMat != 0 && prevMat != matV[0]) break;

	vec3 blockRelPos[3];
	float sizeHeuristic = sqrt(area);
	int mostPerpendicularAxis = 0;
	for (int i = 1; i < 3; i++) {
		vxPos[i] -= 0.01 * sizeHeuristic * normal;
		blockRelPos[i] = vxPos[i] - correspondingBlock;
		if (normal[i] > normal[mostPerpendicularAxis]) {
			mostPerpendicularAxis = i;
		}
	}

	int side = 0;
	for (int k = 0; k < 6; k++) {
		vec3 dir = mat3(k/3 * 2 - 1)[k%3];
		if (dot(dir, normal) > 0.99 && dot(dir, center - correspondingBlock - 0.5) > 0.48) {
			side = k+1;
		}
	}
	if (!claimMaterial(matV[0], side, blockCoords)) break;

	vec2[3] projectedPos;
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			projectedPos[i][j] =
				blockRelPos[i][(mostPerpendicularAxis + j + 1) % 3];
		}
	}
	mat3 localToProjected =
		mat3(vec3(projectedPos[1] - projectedPos[0], 0),
			vec3(projectedPos[2] - projectedPos[0], 0),
			vec3(projectedPos[0], 1));
	mat3 projectedToLocal = inverse(localToProjected);
	vec3 depthFromProjected =
		vec3(blockRelPos[1][mostPerpendicularAxis] -
			blockRelPos[0][mostPerpendicularAxis],
			blockRelPos[2][mostPerpendicularAxis] -
			blockRelPos[0][mostPerpendicularAxis],
			blockRelPos[0][mostPerpendicularAxis]);

	mat3 textureMatrix = mat3(vec3(texCoordV[1] - texCoordV[0], 0),
							vec3(texCoordV[2] - texCoordV[0], 0),
							vec3(texCoordV[0], 1));

	ivec3 boundCoords[2] = ivec3[2](
		ivec3(max(vec3(0), (bounds[0] - correspondingBlock)) * (1<<voxelDetailAmount)),
		ivec3(min(vec3(0.9999), (bounds[1] - correspondingBlock)) * (1<<voxelDetailAmount))
	);

	ivec2 projectionBoundCoords[2];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			projectionBoundCoords[i][j] =
				boundCoords[i]
							[(mostPerpendicularAxis + j + 1) % 3];
		}
	}

	for (int x = projectionBoundCoords[0].x;
			x <= projectionBoundCoords[1].x; x++) {
		for (int y = projectionBoundCoords[0].y;
				y <= projectionBoundCoords[1].y; y++) {
			vec3 thisPos;
			for (int i = 1; i < 3; i++) {
				thisPos[(i + mostPerpendicularAxis) % 3] =
					vec3(0, x + 0.5, y + 0.5)[i];
			}
			thisPos =
				(thisPos - voxelVolumeSize * 0.5) / (1 << voxelDetailAmount);
			vec3 thisProjectedPos =
				vec3(thisPos[(mostPerpendicularAxis + 1) % 3],
						thisPos[(mostPerpendicularAxis + 2) % 3], 1);
			vec3 localPos = projectedToLocal * thisProjectedPos;
			if (localPos.x < 0 || localPos.y < 0 ||
				localPos.x + localPos.y > 1) {
				continue;
			}
			thisPos[mostPerpendicularAxis] =
				dot(depthFromProjected, thisProjectedPos);
			vec2 thisTexCoord = (textureMatrix * localPos).xy;
			vec4 color = texture(tex, thisTexCoord);
			if (color.a < 0.1) {
				continue;
			}
			voxel_t voxelData;
			voxelData.color = color;
			voxelData.full = firstValidLod;
			voxelData.emissive = isEmissive(mat);
			writeVoxelVolume(vxPosToVxCoords(thisPos, voxelDetailAmount), voxelDetailAmount,
								voxelData);
		}
	}
} while (false);