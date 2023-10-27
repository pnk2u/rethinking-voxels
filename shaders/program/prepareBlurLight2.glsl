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
	uniform vec3 cameraPosition;
	#define DENOISE_DATA
	#include "/lib/vx/SSBOs.glsl"
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

		vec2 gradient = vec2(0);

		for (int k = 0; k < 4; k++) {
			ivec2 offset = ivec2(k-1, k-2) % 2 * 3;
			vec4 aroundNormalDepthData = texelFetch(colortex8, texelCoord + 2 * offset, 0);
			aroundNormalDepthData.w = 50.0 * GetLinearDepth(1 - aroundNormalDepthData.w);
			vec4 aroundPreBlurredData0 = texelFetch(colortex10, preBlurredCoord + offset, 0);
			if (length(aroundNormalDepthData - normalDepthData) < 0.1) gradient += offset * (aroundPreBlurredData0.y - thisPreBlurredData0.y);
		}
		float gradientLen2 = dot(gradient, gradient);
		vec2 gradientOffset = min(100 * gradientLen2, 1) * 6 * thisPreBlurredData0.y * gradient / max(gradientLen2, 0.001);
		vec4 offsetPreBlurredData0 = texelFetch(colortex10, preBlurredCoord - ivec2(gradientOffset), 0);
		float offsetBrightness = dot(offsetPreBlurredData0.xyz, offsetPreBlurredData0.xyz);

		float accumulationAmount = thisLightData.a;
		float variance = denoiseSecondMoment[
			texelCoord.x + 
			int(viewWidth + 0.5) * (
				texelCoord.y + int(viewHeight + 0.5) * (frameCounter % 2)
			)
		] - pow2(dot(thisLightData.rgb, vec3(1)));

		variance = clamp(variance, 0.07 - 0.2 * accumulationAmount, 0.9);

		int blurSize = DENOISE_MAX_BLUR_MOD;/*int(
			(DENOISE_MAX_BLUR_MOD - max(DENOISE_MAX_BLUR_MOD - DENOISE_MIN_BLUR_MOD, 0) * 
			min(gradientLen2, accumulationAmount * 2)) * 
			(1.0 + DENOISE_CONVERGED_MULT - accumulationAmount));*/
		if (blurSize < 1) blurSize = 1;
	#else
		int blurSize = int(thisLightData.a + 0.02);
		float variance = fract(thisLightData.a) - 0.05;
	#endif
	variance = max(variance, 0.00001);
	float totalWeight = 0.00001;
	vec3 totalLight = vec3(0.0);
	int blurSize1 = int(sqrt(blurSize));
	#ifndef FIRST
		blurSize1 = blurSize / blurSize1;
	#endif
	blurSize1 *= 2;
	for (int k = 0; k < blurSize1 * blurSize1; k++) {
		//ivec2 offset = ivec2(0);
		//offset.BLUR_AXIS += k;
		ivec2 offset = 2 * ivec2(k%blurSize1, k/blurSize1) - blurSize1;
		#ifndef FIRST
			offset *= blurSize1/2;
		#endif
		if (any(lessThan(offset, -texelCoord + readBounds[0])) || any(greaterThan(offset, readBounds[1] - texelCoord))) {
			continue;
		};
		vec4 aroundNormalDepthData = texelFetch(colortex8, texelCoord + offset, 0);
		aroundNormalDepthData.w = 50.0 * GetLinearDepth(1 - aroundNormalDepthData.w);
		vec4 aroundLight = texelFetch(LIGHT_SAMPLER, texelCoord + offset, 0);
		float weight = exp(-dot(offset, offset) * (2.0 / (blurSize * blurSize)) - pow2(dot(aroundLight.rgb - thisLightData.rgb, vec3(1))) / (4 * variance)) * max0(1 - 7 * length(normalDepthData - aroundNormalDepthData));
		totalWeight += weight;
		totalLight += aroundLight.xyz * weight;
	}
	/*RENDERTARGETS:13*/
	gl_FragData[0] = vec4(totalLight / totalWeight, blurSize + variance + 0.05);
//	#ifdef FIRST
//	gl_FragData[0] = vec4(0.02 * blurSize, 0, 0, 1);
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