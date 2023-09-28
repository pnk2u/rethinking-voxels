#include "/lib/common.glsl"
#ifdef FSH

uniform int frameCounter;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform float near;
uniform float far;
float farPlusNear = far + near;
float farMinusNear = far - near;

uniform sampler2D colortex8;
uniform sampler2D colortex10;
uniform sampler2D colortex13;
#ifdef FIRST
	uniform sampler2D colortex12;
	#define LIGHT_SAMPLER colortex12
#else
	#define LIGHT_SAMPLER colortex13
#endif

float GetLinearDepth(float depth) {
	return (2.0 * near) / (farPlusNear - depth * (farMinusNear));
}

ivec2[2] readBounds = ivec2[2](ivec2(0), ivec2(view));

void main() {
	ivec2 texelCoord = ivec2(gl_FragCoord.xy);
	vec4 prevTex13Data = texelFetch(colortex13, texelCoord, 0);

	vec4 normalDepthData = texelFetch(colortex8, texelCoord, 0);
	normalDepthData.w = 50.0 * GetLinearDepth(1 - normalDepthData.w);
	vec4 thisLightData = texelFetch(LIGHT_SAMPLER, texelCoord, 0);
	#ifdef FIRST
		ivec2 preBlurredCoord = texelCoord / 2 + ivec2(view.x / 2, 0);
		float variance = 0;
		vec4 thisPreBlurredData = texelFetch(colortex10, preBlurredCoord, 0);
		float brightness = dot(thisPreBlurredData.xyz, thisPreBlurredData.xyz);
		vec4[4] preBlurredData;
		vec4[4] varianceRejectionNormalDepthData;
		for (int k = 0; k < 4; k++) {
			ivec2 offset = (2 * (k/2%2) - 1) * ivec2(k%2, (k+1)%2);
			preBlurredData[k] = texelFetch(colortex10, preBlurredCoord + 2 * offset, 0);
			varianceRejectionNormalDepthData[k] = texelFetch(colortex8, texelCoord + 4 * offset, 0);
			varianceRejectionNormalDepthData[k].w = 50.0 * GetLinearDepth(1 - varianceRejectionNormalDepthData[k].w);
		}
		for (int k = 0; k < 4; k++) {
			if (length(varianceRejectionNormalDepthData[k] - normalDepthData) < 0.3) {
				vec3 diff = thisPreBlurredData.xyz - preBlurredData[k].xyz;
				brightness = max(brightness, dot(preBlurredData[k].xyz, preBlurredData[k].xyz));
				variance += dot(diff, diff);
			}
		}
		variance = clamp(sqrt(variance) * 4 / (brightness + 0.05) - 0.2, 0, 1);
		int blurSize = int((DENOISE_MAX_BLUR - max(DENOISE_MAX_BLUR - DENOISE_MIN_BLUR, 0) * variance) * (1.0 + DENOISE_CONVERGED_MULT - fract(thisLightData.a)));
		if (blurSize < 1) blurSize = 1;
	#else
		int blurSize = int(thisLightData.w + 0.5);
	#endif
	float totalWeight = 0.00001;
	vec3 totalLight = vec3(0.0);
	for (int k = -blurSize; k <= blurSize; k++) {
		ivec2 offset = ivec2(0);
		offset.BLUR_AXIS += k;
		if (any(lessThan(offset, -texelCoord + readBounds[0])) || any(greaterThan(offset, readBounds[1] - texelCoord))) {
			continue;
		};
		vec4 aroundNormalDepthData = texelFetch(colortex8, texelCoord + offset, 0);
		aroundNormalDepthData.w = 50.0 * GetLinearDepth(1 - aroundNormalDepthData.w);
		vec4 aroundLight = texelFetch(LIGHT_SAMPLER, texelCoord + offset, 0);
		float weight = exp(-k*k * (2.0 / (blurSize * blurSize))) * max0(1 - 7 * length(normalDepthData - aroundNormalDepthData));
		totalWeight += weight;
		totalLight += aroundLight.xyz * weight;
	}
	/*RENDERTARGETS:13*/
//	#ifdef FIRST
//	gl_FragData[0] = vec4(vec3(variance), 1);
//	#else
	gl_FragData[0] = vec4(totalLight / totalWeight, blurSize + prevTex13Data.a);
//	#endif
}
#endif
#ifdef VSH
void main() {
	gl_Position = ftransform();
}
#endif