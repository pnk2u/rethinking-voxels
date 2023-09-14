#ifndef RAYTRACE
#define RAYTRACE
struct ray_hit_t {
	vec4 rayColor;
	vec4 hitColor;
	vec3 pos;
	vec3 normal;
	int mat;
	vec3 transPos;
	vec3 transNormal;
	vec4 transColor;
	int transMat;
};

struct raytrace_state_t {
	vec3 start;
	vec3 dir;
	vec3 dirSgn;
	mat3 eyeOffsets;
	float rayOffset;
	float rayLength;
	vec3 stepSize;
	vec3 progress;
	int normal;
	float w;
	bool insideVolume;
};

void handleVoxel(inout raytrace_state_t state,
                 inout ray_hit_t returnVal) {
	vec3 pos = state.start + state.w * state.dir;
	ivec3 globalCoord = vxPosToVxCoords(pos + state.eyeOffsets[state.normal]);
	if (globalCoord == ivec3(-1)) {
		state.insideVolume = false;
		return;
	}
	int thisVoxelMat = int(readBlockVolume(globalCoord));
	if (thisVoxelMat == 0) {
		return;
	}
	int glColor0 = readGlColor(globalCoord);
	vec3 glColor = vec3(glColor0 & 255, glColor0 >> 8 & 255, glColor0 >> 16 & 255) / 255.0;
	vec3 baseBlock = floor(pos + state.eyeOffsets[state.normal]);
	pos -= baseBlock;
	int baseIndex = getBaseIndex(thisVoxelMat);
	const int lodResolution = 1<<(VOXEL_DETAIL_AMOUNT-1);
	vec3 localProgress = state.progress;
	vec3 localStepSize = state.stepSize / lodResolution;
	int localNormal = state.normal;
	for (int i = 0; i < 3; i++) {
		for (; state.w < localProgress[i] - localStepSize[i] - 0.000001;
			localProgress[i] -= localStepSize[i]);
	}

	for (int k = 0;
		floor(pos + state.eyeOffsets[localNormal]) == vec3(0) && k < 3 * (1 << VOXEL_DETAIL_AMOUNT-1);
		k++) {
		ivec3 coords = ivec3(lodResolution * pos + state.eyeOffsets[localNormal]);
		voxel_t thisVoxel = readGeometry(baseIndex, coords);
		if (thisVoxel.glColored) {
			thisVoxel.color.rgb *= glColor;
		}
		returnVal.rayColor += (1 - returnVal.rayColor.a) *
								thisVoxel.color.a *
								vec4(thisVoxel.color.rgb, 1);
		if (thisVoxel.color.a > 0.7) {
			returnVal.normal = -state.dirSgn[localNormal] * mat3(1)[localNormal];
		}
		localProgress[localNormal] += localStepSize[localNormal];
		if (returnVal.rayColor.a > 0.99) {
			break;
		}
		localNormal = 0;
		state.w = localProgress[0];
		for (int i = 1; i < 3; i++) {
			if (localProgress[i] < localProgress[localNormal]) {
				localNormal = i;
				state.w = localProgress[localNormal];
			}
		}
		pos = state.start + state.w * state.dir - baseBlock;
	}
}

ray_hit_t raytrace(vec3 start, vec3 dir) {
	ray_hit_t returnVal;
	returnVal.pos = start + dir;
	returnVal.normal = vec3(0, 0, 0);
	returnVal.rayColor = vec4(0);
	returnVal.hitColor = vec4(0);
	returnVal.transColor = vec4(0);
	returnVal.rayColor = vec4(0);
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
	state.eyeOffsets = 1e-2 * mat3(state.dirSgn.x, 0, 0, 0, state.dirSgn.y, 0, 0, 0, state.dirSgn.z);
	// next intersection along each axis
	state.progress =
	    (0.5 + 0.5 * state.dirSgn - fract(state.start)) / state.dir;
	// handle voxel at starting position
	state.normal = 0;
	state.w = state.rayOffset;
	handleVoxel(state, returnVal);
	if (returnVal.rayColor.a > 0.99) {
		return returnVal;
	}
	// closest upcoming intersection
	state.normal = 0;
	state.w = state.progress[0];
	for (int i = 1; i < 3; i++) {
		if (state.progress[i] < state.w) {
			state.normal = i;
			state.w = state.progress[i];
		}
	}
	for (int k = 0;
	     state.w < 1 && k < 2000;
	     k++) {
		int outerNormal = state.normal;
		handleVoxel(state, returnVal);
		if (returnVal.rayColor.a > 0.99 || !state.insideVolume) {
			break;
		}
		state.progress[outerNormal] += state.stepSize[outerNormal];
		state.w = state.progress[0];
		state.normal = 0;
		for (int i = 1; i < 3; i++) {
			if (state.progress[i] < state.w) {
				state.normal = i;
				state.w = state.progress[i];
			}
		}
	}
	returnVal.pos = state.start + state.w * state.dir;
	return returnVal;
}
#endif
