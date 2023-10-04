#include "/lib/colors/lightAndAmbientColors.glsl"

uniform sampler2D colortex14;

#if CONWAY == 1
	const vec4 cylinderColor0 = vec4(0.9, 0.9, 0.9, 3.0);
	const vec4 cylinderColor1 = vec4(0.4, 0.4, 0.4, 3.0);
#else
	const vec4 cylinderColor0 = vec4(0.8 * endLightColor, 3.0);
	const vec4 cylinderColor1 = vec4(0.8 * vec3(1.0, 0.3, 0.0), 3.0);
#endif
// hash without sin by david hoskins (https://www.shadertoy.com/view/XdGfRR)
#define UI0 1597334673U
#define UI1 3812015801U
#define UI2 uvec2(UI0, UI1)
#define UIF (1.0 / float(0xffffffffU))

float hash12(vec2 p)
{
	uvec2 q = uvec2(ivec2(p)) * UI2;
	uint n = (q.x ^ q.y) * UI0;
	return float(n) * UIF;
}

vec4 GetConway(vec3 translucentMult, vec3 playerPos, float dist0, float dist1, float dither) {
    if (min(cameraPosition.y, playerPos.y + cameraPosition.y) > CONWAY_HEIGHT) {
        return vec4(0);
    }
    float slopeFactor0 = playerPos.y / length(playerPos.xz);

    float slopeFactor = sqrt(1 + slopeFactor0 * slopeFactor0);
    playerPos = vec3(0.25, 1.0, 0.25) * normalize(playerPos) * dist1;
    playerPos += 0.0002 * vec3(lessThan(abs(playerPos), vec3(0.0001)));
    float lPlayerPos = length(playerPos.xz);
    vec2 dir = playerPos.xz / lPlayerPos;
    vec3 start = vec3(fract(0.25 * cameraPosition.xz), cameraPosition.y).xzy;
    vec2 stepSize = 1.0 / abs(playerPos.xz);
    vec2 progress = (vec2(greaterThan(playerPos.xz, vec2(0))) - start.xz) / playerPos.xz - vec2(stepSize.x, 0);
    vec4 color = vec4(0);
    float startInVolume = start.y > CONWAY_HEIGHT ? max((CONWAY_HEIGHT - start.y) / playerPos.y, 0.0) : 0.0;
    float stopInVolume = start.y < CONWAY_HEIGHT ? (CONWAY_HEIGHT - start.y) / playerPos.y : 1.0;
    stopInVolume = stopInVolume < 0.0 ? 1.0 : min(stopInVolume, 1.0);
    stopInVolume = playerPos.y < 0 ? min(stopInVolume, (CONWAY_HEIGHT - 30 - start.y) / playerPos.y) : stopInVolume;
    float rayOffset = 0.001 / lPlayerPos;
    float w = startInVolume;
    progress += floor((startInVolume - progress) / stepSize) * stepSize;
    for (; w < stopInVolume; w = min(progress.x, progress.y)) {
        vec2 mask = vec2(lessThanEqual(progress.xy, progress.yx));
        progress += mask * stepSize;
        float nextW = min(progress.x, progress.y);
        vec3 pos = start + playerPos * (w + rayOffset);
        float livelihood = texelFetch(colortex14, ivec2(floor(pos.xz) + 0.1 + vec2(viewWidth, viewHeight) * 0.5), 0).r;
        vec2 circleCenter = floor(pos.xz) + 0.5 - start.xz;
        float circleW = dot(dir, circleCenter) / lPlayerPos;
        float dist = length(playerPos.xz * circleW - circleCenter);
        float insideLen = 2.0 * sqrt((1.0 / 9.0) - dist * dist) / lPlayerPos;
        if (insideLen != insideLen) {
            insideLen = -(1.0 / 3.0) / lPlayerPos;
        }
        float onset = max(circleW - 0.5 * insideLen, startInVolume);
        float offset = min(circleW + 0.5 * insideLen, stopInVolume);
        insideLen = offset - onset;
        if (offset > onset && onset < 1.0) {
			float starty = start.y + playerPos.y * onset - CONWAY_HEIGHT;
			float stopy = start.y + playerPos.y * offset - CONWAY_HEIGHT;
			float cylinderFactor = exp(0.3 * max(starty, stopy)) * livelihood;
			float baseCylinderDensity = cylinderColor0.a * cylinderFactor;
			float cylinderDensity = 1 - exp(-baseCylinderDensity * insideLen * dist1);
			color += cylinderDensity * vec4((circleW > dist0 / dist1 ? translucentMult : vec3(1)) * mix(cylinderColor0.rgb, cylinderColor1.rgb, cylinderFactor * 0.5 + 0.4 * hash12(floor(pos.xz) + 0.25 * cameraPosition.xz) - 0.2) * cylinderFactor, 1.0) * (1 - color.a);
			if (color.a > 0.999) {
				break;
			}
        }
    }
    return color;
}