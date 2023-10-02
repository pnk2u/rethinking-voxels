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
#ifdef FIRST
	uniform sampler2D colortex12;
	#define LIGHT_SAMPLER colortex12
#else
	uniform sampler2D colortex10;
	#define LIGHT_SAMPLER colortex10
#endif

float GetLinearDepth(float depth) {
	return (2.0 * near) / (farPlusNear - depth * (farMinusNear));
}
#ifndef FIRST
	ivec2[2] readBounds = ivec2[2](ivec2(view.x / 2.0, 0), ivec2(view.x, view.y / 2.0));
#else
	ivec2[2] readBounds = ivec2[2](ivec2(0), ivec2(view));
#endif
void main() {
	ivec2 screenTexelCoord = (ivec2(gl_FragCoord.xy) - ivec2(view.x / 2, 0)) * 2;

	#ifdef FIRST
		ivec2 lightingTexelCoord = screenTexelCoord;
	#else
		ivec2 lightingTexelCoord = ivec2(gl_FragCoord.xy);
	#endif
	vec4 normalDepthData = texelFetch(colortex8, screenTexelCoord, 0);
	normalDepthData.w = 50.0 * GetLinearDepth(1 - normalDepthData.w);
	vec4 thisLightData = texelFetch(LIGHT_SAMPLER, lightingTexelCoord, 0);
	#ifdef FIRST
		int blurSize = 18 - int(8 * thisLightData.a);
	#else
		int blurSize = int(thisLightData.a + 0.5);
	#endif
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
		#ifndef FIRST
			offset /= 2;
		#endif
		vec4 aroundLight = texelFetch(LIGHT_SAMPLER, lightingTexelCoord + offset, 0);
		float weight = exp(-k*k * (2.0 / (blurSize * blurSize))) * max0(1 - length(normalDepthData - aroundNormalDepthData));
		totalWeight += weight;
		totalLight += aroundLight.xyz * weight;
	}
	/*RENDERTARGETS:10*/
	gl_FragData[0] = vec4(totalLight / totalWeight, blurSize);
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
	gl_Position /= gl_Position.w;
	gl_Position.xy = 0.5 * gl_Position.xy + vec2(0.5, -0.5);
}
#endif