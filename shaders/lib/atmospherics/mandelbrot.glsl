#ifndef INCLUDE_MANDELBROT
#define INCLUDE_MANDELBROT

#define MANDELBROT_MAX_ITER 100
int mandelbrot(vec2 coords) {
	float p = pow2(coords.x - 0.25) + coords.y * coords.y;
	if (p * (p + coords.x - 0.25) <= 0.25 * coords.y * coords.y || pow2(coords.x - 1) + coords.y * coords.y <= 0.0625) {
		return -1;
	}
	vec2 z = coords;
	int iter = -1;
	for (int k = 0; k < MANDELBROT_MAX_ITER; k++) {
		z = vec2(z.x * z.x - z.y * z.y, 2 * z.x * z.y) + coords;
		if (length(z) > 2.0) {
			iter = k+1;
			break;
		}
	}
	return iter;
}

vec4 mandelbrotColorMap(float iter) {
	return vec4(iter);
}
void mandelbrotSkyColorMod(inout vec3 skyColor, vec3 dir, vec3 sunDir) {
	mat3 sunRotMat0 = mat3(-sunDir, normalize(cross(sunDir, vec3(0, 0, 1))), vec3(0));
	sunRotMat0[2] = cross(sunRotMat0[0], sunRotMat0[1]);
	mat3 sunRotMat = inverse(sunRotMat0);
	dir = sunRotMat * dir;
	if (dir.x < 0) {
		return;
	}
	vec2 coords = (2 - dir.x) * dir.yz;
	int iter_count = mandelbrot(coords);
	vec4 mandelbrotColor = mandelbrotColorMap(iter_count * 1.0 / MANDELBROT_MAX_ITER);
	skyColor = mix(skyColor, mandelbrotColor.rgb, mandelbrotColor.a);
}
#endif