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
		float accumulationAmount = thisLightData.a;
		float variance = denoiseSecondMoment[
			texelCoord.x + 
			int(viewWidth + 0.5) * (
				texelCoord.y + int(viewHeight + 0.5) * (frameCounter % 2)
			)
		] - pow2(dot(thisLightData.rgb, vec3(1)));
		variance /= max(0.01, accumulationAmount / ACCUM_FALLOFF_SPEED);
		variance = max(variance, 0.07 - 0.2 * accumulationAmount);

	#else
		float variance = thisLightData.a;
	#endif
	variance = max(variance, 0.00001);
	float totalWeight = 0.00001;
	vec3 totalLight = vec3(0.0);
	int blurSize1 = int(sqrt(DENOISE_MAX_BLUR_MOD));
	#ifdef FIRST
		blurSize1 = DENOISE_MAX_BLUR_MOD / blurSize1;
	#endif
	blurSize1 *= 2;
	for (int k = 0; k < blurSize1 * blurSize1; k++) {

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
		float weight = exp(-dot(offset, offset) * (2.0 / (DENOISE_MAX_BLUR_MOD * DENOISE_MAX_BLUR_MOD)) - pow2(dot(aroundLight.rgb - thisLightData.rgb, vec3(1))) / (4 * variance)) * max0(1 - 7 * length(normalDepthData - aroundNormalDepthData));
		totalWeight += weight;
		totalLight += aroundLight.xyz * weight;
	}
	/*RENDERTARGETS:13*/
	gl_FragData[0] = vec4(totalLight / totalWeight, variance);
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