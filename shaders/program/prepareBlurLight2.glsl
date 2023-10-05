#include "/lib/common.glsl"
#ifdef FSH
#ifdef DENOISING
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
#endif
uniform sampler2D colortex13;
#ifdef FIRST
	uniform sampler2D colortex12;
	#ifdef DENOISING
		#define LIGHT_SAMPLER colortex12
	#endif
#elif defined DENOISING
	#define LIGHT_SAMPLER colortex13
#endif
#ifdef DENOISING
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
		vec4 thisPreBlurredData0 = texelFetch(colortex10, preBlurredCoord, 0);
		vec4 thisPreBlurredData1 = texelFetch(colortex10, preBlurredCoord + ivec2(0, view.y / 2.0 + 0.1), 0);
		float brightness = dot(thisPreBlurredData0.xyz, thisPreBlurredData0.xyz);
		float variance0 = length(thisLightData.rgb - thisPreBlurredData0.rgb);
		float variance1 = length(thisPreBlurredData0.rgb - thisPreBlurredData1.rgb);
		float variance = clamp(variance0 * variance1 * 50 / (brightness + 0.02), 0, 1);
		float accumulationAmount = fract(thisLightData.a);
		int blurSize = int((DENOISE_MAX_BLUR - max(DENOISE_MAX_BLUR - DENOISE_MIN_BLUR, 0) * min(variance, accumulationAmount * 2)) * (1.0 + DENOISE_CONVERGED_MULT - accumulationAmount));
		if (blurSize < 1) blurSize = 1;
	#else
		int blurSize = int(thisLightData.w + 0.02);
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
	gl_FragData[0] = vec4(totalLight / totalWeight, blurSize + fract(prevTex13Data.a + 0.05) - 0.05);
//	#ifdef FIRST
//	gl_FragData[0] = vec4(vec3(variance), 1);
//	#endif
}
#elif defined FIRST
void main() {
	ivec2 texelCoord = ivec2(gl_FragCoord.xy);
	vec4 prevTex13Data = texelFetch(colortex13, texelCoord, 0);
	/*RENDERTARGETS:13*/
	gl_FragData[0] = vec4(texelFetch(colortex12, texelCoord, 0).rgb, prevTex13Data.a);
}
#else
	void main() {
		discard;
	}
#endif
#endif
#ifdef VSH
void main() {
	gl_Position = ftransform();
}
#endif