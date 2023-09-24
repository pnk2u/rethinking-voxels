#include "/lib/common.glsl

#ifdef CSH

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform int frameCounter;
uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousProjection;
uniform mat4 gbufferPreviousModelView;
uniform sampler2D colortex8;
layout(rgba16f) uniform image2D colorimg10;
layout(rgba16f) uniform image2D colorimg11;

#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/raytrace.glsl"

uint globalSeed = uint((frameCounter * 100 + gl_GlobalInvocationID.x % 100) * 113 + gl_GlobalInvocationID.y % 113);

uint murmur(uint seed) {
    seed = (seed ^ (seed >> 16)) * 0x85ebca6bu;
    seed = (seed ^ (seed >> 13)) * 0xc2b2ae35u;
    return seed ^ (seed >> 16);
}

uint nextUint() {
    return murmur(globalSeed += 0x9e3779b9u);
}

float nextFloat() {
    return float(nextUint()) / float(uint(0xffffffff));
}

vec3 randomSphereSample() {
	float x1, x2;
	float len2;
	do {
		x1 = nextFloat() * 2 - 1;
		x2 = nextFloat() * 2 - 1;
		len2 = x1 * x1 + x2 * x2;
	} while (len2 >= 1);
	float x3 = sqrt(1 - len2);
	return vec3(
		2 * x1 * x3,
		2 * x2 * x3,
		1 - 2 * len2);
}

#define LIGHT_DISTANCE_FACTOR 0.1
#define MAX_LIGHT_COUNT 16

shared int lightCount = 0;
shared ivec2 prevTexelCoord;
shared vec4[MAX_LIGHT_COUNT] positions;
shared int[MAX_LIGHT_COUNT] mergeOffsets;
shared uvec2 updatedPositions = uvec2(0);

void main() {
	ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
	vec4 normalDepthData = texelFetch(colortex8, texelCoord, 0);
	vec3 vxPosFrameOffset = vec3(ivec3((floor(previousCameraPosition) - floor(cameraPosition)) * 1.1));
	if (normalDepthData.a < 1.5 && length(normalDepthData.rgb) > 0.1) {
		// valid data and not sky
		vec4 playerPos = gbufferModelViewInverse * (gbufferProjectionInverse * (vec4((texelCoord + 0.5) / view, 1 - normalDepthData.a, 1) * 2 - 1));
		playerPos /= playerPos.w;
		if (gl_LocalInvocationID == gl_WorkGroupSize/2) {
			vec4 prevClipPos = gbufferPreviousProjection * (gbufferPreviousModelView * playerPos);
			prevClipPos /= prevClipPos.w;
			prevTexelCoord = ivec2(view * (0.5 * prevClipPos.xy + 0.5)) - ivec2(gl_WorkGroupSize.xy/2);
		}
		vec3 vxPos = playerToVx(playerPos.xyz) + max(0.1, 0.005 * length(playerPos.xyz)) * normalDepthData.xyz;
		vec3 dir = randomSphereSample();
		if (dot(dir, normalDepthData.xyz) < 0) dir *= -1;
		ray_hit_t rayHit0 = raytrace(vxPos, 20 * dir);
		if (rayHit0.emissive) {
			int lightIndex = atomicAdd(lightCount, 1);
			if (lightIndex < MAX_LIGHT_COUNT) {
				positions[lightIndex] = vec4(rayHit0.pos - 0.05 * rayHit0.normal, GetLuminance(rayHit0.rayColor.xyz) * 2.0 / (0.1 + length(rayHit0.pos - vxPos)));
				mergeOffsets[lightIndex] = 0;
			} else {
				atomicMin(lightCount, MAX_LIGHT_COUNT);
			}
		}
		barrier();
		memoryBarrierShared();
		int mergeOffset = 0;
		int oldLightCount = min(lightCount, MAX_LIGHT_COUNT);
		int index = int(gl_LocalInvocationID.x + gl_WorkGroupSize.x * gl_LocalInvocationID.y);
		vec4 thisPos = positions[index];
		ivec3 comparePos = ivec3(thisPos.xyz + 1000);
		int k = index + 1;
		while (k < oldLightCount && ivec3(positions[k].xyz + 1000) != comparePos) k++;
		if (k < oldLightCount) {
			atomicAdd(mergeOffsets[k], -1000);
			mergeOffset = 1;
		}
		for (k++; k < oldLightCount; k++) {
			atomicAdd(mergeOffsets[k], 1);
		}
		barrier();
		memoryBarrierShared();
		if (mergeOffsets[index] > 0 && index < oldLightCount) {
			positions[index - mergeOffsets[index]] = thisPos;
		}
		if (mergeOffset > 0) {
			atomicAdd(lightCount, -1);
		}
		barrier();
		memoryBarrierShared();
		if (lightCount > 0) {
			uint thisLightIndex = nextUint() % lightCount;
			ray_hit_t rayHit1 = raytrace(vxPos, 1.1 * (positions[thisLightIndex].xyz - vxPos));
			imageStore(colorimg10, texelCoord, rayHit1.rayColor * float(rayHit1.emissive));
		}
	}
}
#endif