#include "/lib/common.glsl

#ifdef CSH

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
const vec2 workGroupsRender = vec2(1.0, 1.0);

uniform int frameCounter;
uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);
uniform vec3 cameraPosition;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform sampler2D colortex8;
layout(rgba16f) uniform image2D colorimg10;

#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/raytrace.glsl"

uint globalSeed = uint((frameCounter * uint(view.x) + gl_GlobalInvocationID.x) * uint(view.y) + gl_GlobalInvocationID.y);

uint murmur(uint seed) {
    seed = (seed ^ (seed >> 16)) * 0x85ebca6bu;
    seed = (seed ^ (seed >> 13)) * 0xc2b2ae35u;
    return seed ^ (seed >> 16);
}

uint nextUint() {
    return murmur(globalSeed += 0x9e3779b9u);
}

float nextFloat() {
    return float(nextUint()) / float(0xffffffffu);
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

shared int lightCount = 0;
shared vec3[64] positions;

void main() {
	ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
	vec4 normalDepthData = texelFetch(colortex8, texelCoord, 0);
	if (normalDepthData.a < 1.5 && length(normalDepthData.rgb) > 0.1) {
		// valid data and not sky
		vec4 playerPos = gbufferModelViewInverse * (gbufferProjectionInverse * (vec4((texelCoord + 0.5) / view, 1 - normalDepthData.a, 1) * 2 - 1));
		playerPos /= playerPos.w;
		vec3 vxPos = playerToVx(playerPos.xyz) + max(0.1, 0.005 * length(playerPos.xyz)) * normalDepthData.xyz;
		vec3 dir = randomSphereSample();
		if (dot(dir, normalDepthData.xyz) < 0) dir *= -1;
		ray_hit_t rayHit0 = raytrace(vxPos, 20 * dir);
		if (rayHit0.emissive) {
			int lightIndex = atomicAdd(lightCount, 1);
			positions[min(lightIndex, 63)] = rayHit0.pos;
		}
		barrier();
		memoryBarrierShared();
		if (lightCount == 0) {
			return;
		}
		uint lightIndex = nextUint() % lightCount;
		vec3 nextDir = 1.01 * (positions[lightIndex] - vxPos);
		if (dot(nextDir, normalDepthData.rgb) < 0) {
			return;
		}
		ray_hit_t rayHit1 = raytrace(vxPos, nextDir);
		imageStore(colorimg10, texelCoord, rayHit0.rayColor * float(rayHit0.emissive));
	}
}
#endif