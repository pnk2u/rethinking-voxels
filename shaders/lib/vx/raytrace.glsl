#ifndef RAYTRACE
#define RAYTRACE
struct ray_hit_t {
	vec4 rayColor;
	vec4 hitColor;
	vec3 pos;
	vec3 normal;
	int mat;
	bool emissive;
	vec3 transPos;
	vec3 transNormal;
	vec4 transColor;
	int transMat;
};

struct raytrace_state_t {
	vec3 start;
	vec3 dir;
	vec3 dirSgn;
	vec3 eyeOffsets;
	float rayOffset;
	float rayLength;
	vec3 stepSize;
	vec3 progress;
	vec3 normal;
	float w;
	bool insideVolume;
};
#define MAX_RAY_ALPHA 0.999
void handleVoxel(inout raytrace_state_t state,
                 inout ray_hit_t returnVal) {
	vec3 pos = state.start + state.w * state.dir;
	vec3 normalOffsets = state.eyeOffsets * state.normal;
	ivec3 globalCoord = vxPosToVxCoords(pos + normalOffsets);
	int thisVoxelMat = globalCoord != ivec3(-1) ? int(readBlockVolume(globalCoord)) : 0;
	float entityW = 2;
	int entityOccupancy = readEntityOccupancy(globalCoord);

	vec3 entityNormal = vec3(0);
	vec3 entityCol = vec3(0);
	bool entityEmissive = false;
	for (int _safdghlknef = 0; _safdghlknef == 0 && entityOccupancy != 0; _safdghlknef++) {
		int index = int(
			dot(floor(fract(pos + normalOffsets) + 0.5), vec3(1.01, 2.01, 4.01)));
		if (((entityOccupancy >> index) & 1) != 0) {
			entityW = state.w;
			entityNormal = state.normal;
			entityEmissive = bool(entityOccupancy >> (index + 8) & 1);
			break;
		}
		vec3 entityIsctWs = (floor(pos + normalOffsets) + 0.5 - state.start) / state.dir;

		for (int k = 0; k < 3; k++) {
			if (entityIsctWs[k] > min(entityW, 1) || entityIsctWs[k] < state.w - state.rayOffset) continue;
			vec3 thisNormal = vec3(k == 0, k == 1, k == 2);
			vec3 isctPos = state.start + entityIsctWs[k] * state.dir +
			state.eyeOffsets * thisNormal;
			index = int(dot(floor(fract(isctPos) + 0.5), vec3(1.01, 2.01, 4.01)));
			if (length(floor(isctPos) - floor(pos + normalOffsets)) > 0.5 || ((entityOccupancy >> index) & 1) == 0) {
				continue;
			}
			entityW = entityIsctWs[k];
			entityNormal = thisNormal;
			entityEmissive = bool(entityOccupancy >> (index + 8) & 1);
		}
	}
	if (entityW < 1.0) {
		entityCol = readEntityColor(globalCoord);
		if (thisVoxelMat == 0) {
			state.w = entityW;
			returnVal.rayColor = vec4(entityCol, 1);
			returnVal.emissive = entityEmissive;
			returnVal.normal = state.dirSgn * entityNormal;
			return;
		}
	} else if (thisVoxelMat == 0) {
		return;
	}
	vec3 baseBlock = floor(pos + normalOffsets);
	pos -= baseBlock;
	int baseIndex = getBaseIndex(thisVoxelMat);
	const int lodResolution = 1<<(VOXEL_DETAIL_AMOUNT-1);
	vec3 localProgress = state.progress;
	vec3 localStepSize = state.stepSize / lodResolution;
	vec3 localNormal = state.normal;
	vec3 overshoot = max(floor((localProgress - state.w - state.rayOffset) / abs(localStepSize)), 0);
	localProgress -= overshoot * localStepSize;
	vec3 exitWs = state.progress + localNormal * state.stepSize;
	float exitW = min(min(exitWs[0], exitWs[1]), min(exitWs[2], entityW + 2 * state.rayOffset));
	exitW -= state.rayOffset;
	for (int k = 0; state.w < exitW && k < 3 * (1 << VOXEL_DETAIL_AMOUNT-1); k++) {
		vec3 innerPos = state.start + state.w * state.dir - baseBlock;
		ivec3 coords = ivec3(lodResolution * innerPos + state.eyeOffsets * localNormal);
		voxel_t thisVoxel = readGeometry(baseIndex, coords);
		if (thisVoxel.color.a > 0.1) {
			if (thisVoxel.glColored) {
				int glColor0 = readGlColor(globalCoord);
				vec3 glColor = vec3(glColor0 & 255, glColor0 >> 8 & 255, glColor0 >> 16 & 255) / 255.0;
				thisVoxel.color.rgb *= glColor;
			}
			returnVal.transColor = returnVal.rayColor;
			returnVal.rayColor.rgb *= mix(vec3(1), thisVoxel.color.rgb, thisVoxel.color.a);
			returnVal.rayColor.a += (1 - returnVal.rayColor.a) * thisVoxel.color.a;
			returnVal.emissive = returnVal.emissive || thisVoxel.emissive;
			if (thisVoxel.color.a > 0.9) {
				returnVal.mat = thisVoxelMat;
				returnVal.normal = -state.dirSgn * localNormal;
			} else {
				returnVal.transMat = thisVoxelMat;
				returnVal.transPos = innerPos + baseBlock;
			}
			if (returnVal.rayColor.a > MAX_RAY_ALPHA) {
				return;
			}
		}
		localProgress += localStepSize * localNormal;
		state.w = min(min(localProgress[0], localProgress[1]), localProgress[2]);
		localNormal = vec3(lessThanEqual(localProgress, vec3(state.w)));
	}
	if (entityW < state.w - state.rayOffset) {
		state.w = entityW;
		returnVal.rayColor = vec4(entityCol, 1);
		returnVal.emissive = entityEmissive;
		returnVal.normal = state.dirSgn * entityNormal;
	}
}

ray_hit_t raytrace(vec3 start, vec3 dir) {
	ray_hit_t returnVal;
	returnVal.emissive = false;
	returnVal.pos = start;
	returnVal.normal = vec3(0, 0, 0);
	returnVal.rayColor = vec4(1, 1, 1, 0);
	returnVal.hitColor = vec4(0);
	returnVal.transPos = vec3(-1000);
	returnVal.transColor = vec4(0);
	returnVal.transNormal = vec3(-1);
	returnVal.transMat = -1;
	returnVal.mat = -1;
	raytrace_state_t state;
	state.start = start;
	state.dir = dir + 1e-10 * vec3(equal(dir, vec3(0)));
	state.rayLength = length(state.dir);
	state.stepSize = 1.0 / abs(state.dir);
	state.dirSgn = sign(state.dir);
	state.insideVolume = true;
	// offsets that will be used to avoid floating point
	// errors on block edges
	state.rayOffset = 1e-2 / state.rayLength;
	state.eyeOffsets = 1e-2 * state.dirSgn;
	// next intersection along each axis
	state.progress =
	    (0.5 + 0.5 * state.dirSgn - fract(state.start)) / state.dir;
	// handle voxel at starting position
	state.normal = vec3(0);
	state.w = state.rayOffset;
	handleVoxel(state, returnVal);
	if (returnVal.rayColor.a > MAX_RAY_ALPHA) {
		returnVal.pos = state.start + state.w * state.dir;
		return returnVal;
	}
	// closest upcoming intersection
	state.w = min(min(state.progress[0], state.progress[1]), state.progress[2]);
	state.normal = vec3(lessThanEqual(state.progress, vec3(state.w)));
	for (int k = 0;
	     state.w < 1 && k < 2000;
	     k++) {
		handleVoxel(state, returnVal);
		if (returnVal.rayColor.a > MAX_RAY_ALPHA || !state.insideVolume) {
			break;
		}
		state.progress += state.stepSize * state.normal;
		state.w = min(min(state.progress[0], state.progress[1]), state.progress[2]);
		state.normal = vec3(lessThanEqual(state.progress, vec3(state.w)));
	}
	returnVal.pos = state.start + state.w * state.dir;
	return returnVal;
}
#endif
