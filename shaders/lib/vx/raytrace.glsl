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
	vec3 pos = state.start + state.w * state.dir +
	           state.eyeOffsets[state.normal];
	ivec3 coords[VOXEL_DETAIL_AMOUNT];
	coords[0] = vxPosToVxCoords(pos, 0);
	if (coords[0] == ivec3(-1)) {
		state.insideVolume = false;
		return;
	}
	voxel_t thisVoxel = readVoxelVolume(coords[0], 0);
	if (thisVoxel.full) {
		returnVal.rayColor += (1 - returnVal.rayColor.a) *
		                      thisVoxel.color.a *
		                      vec4(thisVoxel.color.rgb, 1);
		return;
	}
	vec3 subVoxelProgress = state.progress;
	int lod = 1;
	for (int k = 0; lod > 0 && k < 3 * (1 << VOXEL_DETAIL_AMOUNT);
	     k++) {
		coords[lod] = vxPosToVxCoords(pos, lod);
		if (lod < VOXEL_DETAIL_AMOUNT - 1) {
			coords[lod+1] = vxPosToVxCoords(pos, lod);
		}

		thisVoxel = readVoxelVolume(coords[lod], lod);
		if (thisVoxel.full || lod >= VOXEL_DETAIL_AMOUNT - 1 ||
		    coords[lod + 1] == ivec3(-1)) {
			returnVal.rayColor += (1 - returnVal.rayColor.a) *
			                      thisVoxel.color.a *
			                      vec4(thisVoxel.color.rgb, 1);
			subVoxelProgress[state.normal] +=
			    state.stepSize[state.normal] / (1 << lod);
			state.normal = 0;
			state.w = subVoxelProgress[0];
			for (int i = 1; i < 3; i++) {
				if (subVoxelProgress[i] < state.w) {
					state.normal = i;
					state.w = subVoxelProgress[i];
				}
			}
			pos = state.start + state.w * state.dir +
			      state.eyeOffsets[state.normal];
			for (ivec3 parentCoords = vxPosToVxCoords(pos, lod - 1);
			     lod > 0 && (parentCoords != coords[lod - 1]); lod--)
				;
		} else {
			lod++;
		}
	}
}

ray_hit_t raytrace(vec3 start, vec3 dir) {
	ray_hit_t returnVal;
	returnVal.pos = start + dir;
	returnVal.normal = vec3(0, 1, 0);
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
	state.stepSize = 1.0 / state.dir;
	state.dirSgn = sign(state.dir);
	state.insideVolume = true;
	// offsets that will be used to avoid floating point
	// errors on block edges
	state.rayOffset = 1e-3 / state.rayLength;
	state.eyeOffsets = mat3(state.dirSgn.x, 0, 0, 0, state.dirSgn.y, 0, 0, 0, state.dirSgn.z);
	// next intersection along each axis
	state.progress =
	    (fract(state.start) - 0.5 * state.dirSgn - 0.5) / state.dir;
	// handle voxel at starting position
	state.normal = 0;
	state.w = state.rayOffset;
	handleVoxel(state, returnVal);
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
	     state.w < 1 && returnVal.rayColor.a < 0.999 && k < 2000;
	     k++) {
		handleVoxel(state, returnVal);
		state.progress[state.normal] += state.stepSize[state.normal];
		for (int i = 1; i < 3; i++) {
			if (state.progress[i] < state.w) {
				state.normal = i;
				state.w = state.progress[i];
			}
		}
	}
	returnVal.pos = state.start + state.w * state.dir;
	returnVal.normal = -state.dirSgn[state.normal] * mat3(1)[state.normal];
	return returnVal;
}
#endif
