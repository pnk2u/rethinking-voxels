#include "/lib/common.glsl"
#ifdef FSH

uniform int frameCounter;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform sampler2D colortex8;
#ifdef FIRST
	uniform sampler2D colortex10;
	#define LIGHT_SAMPLER colortex10
#else
	uniform sampler2D colortex12;
	#define LIGHT_SAMPLER colortex12
#endif

uniform float near;
uniform float far;
float farPlusNear = far + near;
float farMinusNear = far - near;

float GetLinearDepth(float depth) {
	return (2.0 * near) / (farPlusNear - depth * (farMinusNear));
}
#ifdef FIRST
	ivec2[2] readBounds = ivec2[2](ivec2(0), ivec2(view / 2.0));
#else
	ivec2[2] readBounds = ivec2[2](ivec2(0), ivec2(view));
#endif
void main() {
	ivec2 screenTexelCoord = ivec2(gl_FragCoord.xy);

	#ifdef FIRST
		ivec2 lightingTexelCoord = screenTexelCoord / 2;
	#else
		#define lightingTexelCoord screenTexelCoord
	#endif
	vec4 normalDepthData = texelFetch(colortex8, screenTexelCoord, 0);
	normalDepthData.w = 50.0 * GetLinearDepth(1 - normalDepthData.w);
	vec4 thisLightData = texelFetch(LIGHT_SAMPLER, lightingTexelCoord, 0);
	int blurSize = 30;//int(sqrt(10 * thisLightData.w + 20));
	float totalWeight = 0.00001;
	vec3 totalLight = vec3(0.0);
	for (int k = -blurSize; k <= blurSize; k++) {
		ivec2 offset = ivec2(0);
		offset.BLUR_AXIS += 2 * k;
		if (any(lessThan(offset, -lightingTexelCoord + readBounds[0])) || any(greaterThan(offset, readBounds[1] - lightingTexelCoord))) {
			continue;
		};
		vec4 aroundNormalDepthData = texelFetch(colortex8, screenTexelCoord + offset, 0);
		aroundNormalDepthData.w = 50.0 * GetLinearDepth(1 - aroundNormalDepthData.w);
		#ifdef FIRST
			offset /= 2;
		#endif
		vec4 aroundLight = texelFetch(LIGHT_SAMPLER, lightingTexelCoord + offset, 0);
		float weight = exp(-k*k * (2.0 / (blurSize * blurSize))) * max0(1 - 0.5 * length(normalDepthData - aroundNormalDepthData));
		totalWeight += weight;
		totalLight += aroundLight.xyz * weight;
	}
	/*RENDERTARGETS:12*/
	gl_FragData[0] = vec4(totalLight / totalWeight, blurSize);
}
#endif
#ifdef VSH
void main() {
	gl_Position = ftransform();
}
#endif