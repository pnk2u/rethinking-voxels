#include "/lib/common.glsl"

//////Fragment Shader//////Fragment Shader//////
#ifdef FSH

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform sampler2D colortex2;
uniform sampler2D colortex4;

layout(rgba16f) uniform image2D colorimg8;
layout(r32ui) uniform uimage2D colorimg9;

#define MATERIALMAP_ONLY
#include "/lib/vx/SSBOs.glsl"

void main() {
	float prevDepth = 1 - texelFetch(colortex2, ivec2(gl_FragCoord.xy), 0).w;
	vec4 prevClipPos = vec4(gl_FragCoord.xy / view, prevDepth, 1) * 2 - 1;
	vec4 newClipPos = prevClipPos;
	if (prevDepth > 0.56) {
		newClipPos = reprojectionMatrix * prevClipPos;
		newClipPos /= newClipPos.w;
	}
	newClipPos = 0.5 * newClipPos + 0.5;
	if (prevClipPos.z > 0.99998) newClipPos.z = 0.9999985;
	if (all(greaterThan(newClipPos.xyz, vec3(0))) && all(lessThan(newClipPos.xyz, vec3(0.999999)))) {
		newClipPos.xy *= view;
		vec2 diff = newClipPos.xy - gl_FragCoord.xy + 0.001;
		ivec2 writePixelCoord = ivec2(gl_FragCoord.xy + floor(diff));
		uint depth = uint((1<<30) * newClipPos.z);
		if (imageLoad(colorimg9, writePixelCoord).r == depth) {
			vec2 prevSampleCoord = (gl_FragCoord.xy - fract(diff)) / view;
			vec4 writeData = vec4(newClipPos.z < 0.999998 ? texture(colortex4, prevSampleCoord).gba * 2 - 1 : vec3(0), 1 - newClipPos.z);
			imageStore(colorimg8, writePixelCoord, writeData);
		}
	}
	/*DRAWBUFFERS:3*/
}
#endif

//////Vertex Shader//////Vertex Shader//////
#ifdef VSH

void main() {
	gl_Position = ftransform();
}
#endif