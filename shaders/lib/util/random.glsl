#ifndef INCLUDE_RANDOM
#define INCLUDE_RANDOM

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

#endif