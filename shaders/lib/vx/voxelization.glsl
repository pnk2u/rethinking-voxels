vec3 vxPos[3];
vec3 bounds[2] = vec3[2](vec3(1e10), vec3(-1e10));
for (int i = 0; i < 3; i++) {
	vxPos[i] = playerToVx(positionV[i].xyz);
	bounds[0] = min(bounds[0], vxPos[i]);
	bounds[0] = max(bounds[1], vxPos[i]);
}

vec3 size = bounds[1] - bounds[0];
vec3 center = 0.5 * (bounds[1] + bounds[0]);
for (int i = 0; i < 3; i++) {
	vxPos[i] = mix(vxPos[i], center, 0.01);
}

vec3 normal = cross(vxPos[1] - vxPos[0], vxPos[2] - vxPos[0]);
float area = max(length(normal), 1e-10);
normal /= area;
float sizeHeuristic = sqrt(area);
int mostPerpendicularAxis = 0;
for (int i = 1; i < 3; i++) {
	vxPos[i] -= 0.01 * sizeHeuristic * normal;
	if (normal[i] > normal[mostPerpendicularAxis]) {
		mostPerpendicularAxis = i;
	}
}

vec2[3] projectedPos;
for (int i = 0; i < 3; i++) {
	for (int j = 0; j < 2; j++) {
		projectedPos[i][j] =
		    vxPos[i][(mostPerpendicularAxis + j + 1) % 3];
	}
}
mat3 localToProjected =
    mat3(vec3(projectedPos[1] - projectedPos[0], 0),
         vec3(projectedPos[2] - projectedPos[0], 0),
         vec3(projectedPos[0], 1));
mat3 projectedToLocal = inverse(localToProjected);
vec3 depthFromProjected = vec3(
    vxPos[1][mostPerpendicularAxis] - vxPos[0][mostPerpendicularAxis],
    vxPos[2][mostPerpendicularAxis] - vxPos[0][mostPerpendicularAxis],
    vxPos[0][mostPerpendicularAxis]);

mat3 textureMatrix =
    mat3(vec3(texCoordV[1] - texCoordV[0], 0),
         vec3(texCoordV[2] - texCoordV[0], 0), vec3(texCoordV[0], 1));

bool firstValidLod = true;
for (int lod = voxelDetailAmount - 1; lod >= 0; lod--) {
	ivec3 boundCoords[2] = ivec3[2](vxPosToVxCoords(bounds[0], lod),
	                                vxPosToVxCoords(bounds[1], lod));
	if (boundCoords[0] == ivec3(-1) || boundCoords[1] == ivec3(-1)) {
		continue;
	}
	ivec2 projectionBoundCoords[2];
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			projectionBoundCoords[i][j] =
			    boundCoords[i][(mostPerpendicularAxis + j + 1) % 3];
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
			thisPos = (thisPos - voxelVolumeSize * 0.5) / (1 << lod);
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
			writeVoxelVolume(vxPosToVxCoords(thisPos, lod), lod, voxelData);
		}
	}
	firstValidLod = false;
}
