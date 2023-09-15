for (int _lkakmdffonef = 0; _lkakmdffonef < 1; _lkakmdffonef++) {
	vec3 vxPos[3];
	vec3 bounds[2] = vec3[2](vec3(1e10), vec3(-1e10));
	for (int i = 0; i < 3; i++) {
		vxPos[i] = playerToVx(positionV[i].xyz);
		bounds[0] = min(bounds[0], vxPos[i]);
		bounds[1] = max(bounds[1], vxPos[i]);
	}

	if (matV[0] == 0 || !isInRange(bounds[0]) || !isInRange(bounds[1])) break;
	vec3 normal = cross(vxPos[1] - vxPos[0], vxPos[2] - vxPos[0]);
	float area = max(length(normal), 1e-10);
	normal /= area;

	vec3 center = 0.5 * (bounds[1] + bounds[0]);
	vec3 size = bounds[1] - bounds[0];
	for (int i = 0; i < 3; i++) {
		vxPos[i] = mix(vxPos[i], center, 0.01);
	}

	// center cross-model blocks
	if (abs(normal.y) < 0.1 && abs(abs(normal.x) - abs(normal.z)) < 0.1) {
		for (int i = 0; i < 3; i++) {
			vxPos[i].xz += 0.5 - fract(center.xz);
		}
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

	ivec3 blockCoords = vxPosToVxCoords(correspondingBlock + 0.5);
	uint prevMat = imageAtomicCompSwap(voxelVolumeI, blockCoords, 0, matV[0]);
	if (prevMat != 0 && prevMat != uint(matV[0])) break;

	vec4 meanGlColor = 0.333 * (glColorV[0] + glColorV[1] + glColorV[2]);
	bool hasGlColor = (length(meanGlColor.rgb - vec3(1)) > 0.1);
	if (hasGlColor) {
		int packedGlColor =  int(255.5 * meanGlColor.r)
		                  + (int(255.5 * meanGlColor.g) << 8)
		                  + (int(255.5 * meanGlColor.b) << 16);
		imageStore(voxelVolumeI, blockCoords + ivec3(0, voxelVolumeSize.y, 0), ivec4(packedGlColor));
	}
	vec3 blockRelPos[3];
	float sizeHeuristic = sqrt(area);
	int mostPerpendicularAxis = 0;
	for (int i = 0; i < 3; i++) {
		// get faces away from integer coordinates prone to precision issues
		vxPos[i] -= 0.01 * sizeHeuristic * normal;
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

	int baseIndex = getBaseIndex(matV[0]);
	for (int x = projectionBoundCoords[0].x;
			x <= projectionBoundCoords[1].x; x++) {
		for (int y = projectionBoundCoords[0].y;
				y <= projectionBoundCoords[1].y; y++) {
			vec3 thisProjectedPos = vec3(x + 0.5, y + 0.5, 1) / vec2(1 << (VOXEL_DETAIL_AMOUNT-1), 1).xxy;
			vec3 localPos = projectedToLocal * thisProjectedPos;
			if (localPos.x < -0.001 || localPos.y < -0.001 ||
				localPos.x + localPos.y > 1.001) {
				continue;
			}
			vec2 thisTexCoord = (textureMatrix * localPos).xy;
			vec4 color = textureLod(tex, thisTexCoord, 0);
			vec4 s = textureLod(specular, thisTexCoord, 0);
			if (s.a > 0.999) s.a = 0;
			if (s.a > 0.1)  {
				color.rgb *= s.a;
			}
			vec3 thisPos = projectedToBlockRel * thisProjectedPos;
			if (color.a < 0.1 || thisPos != clamp(thisPos, 0, 1) || badPixel(color, meanGlColor, matV[0])) {
				continue;
			}

			voxel_t voxelData;
			voxelData.color = color;
			voxelData.glColored = hasGlColor;
			voxelData.emissive = (/*isEmissive(blockIdMap[matV[0]]) ||*/ s.a > 0.1);
			writeGeometry(baseIndex, thisPos, voxelData);
		}
	}
}
