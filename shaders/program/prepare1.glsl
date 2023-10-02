#include "/lib/common.glsl"
//////Fragment Shader//////Fragment Shader//////
#ifdef FSH
flat in mat4 localReprojectionMatrix;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform sampler2D colortex2;

layout(r32ui) uniform uimage2D colorimg9;

void main() {
	float prevDepth = 1 - texelFetch(colortex2, ivec2(gl_FragCoord.xy), 0).w;
	vec4 prevClipPos = vec4(gl_FragCoord.xy / view, prevDepth, 1) * 2 - 1;
	vec4 newClipPos = prevClipPos;
	if (prevDepth > 0.56) {
		newClipPos = localReprojectionMatrix * prevClipPos;
		newClipPos /= newClipPos.w;
	}
	newClipPos = 0.5 * newClipPos + 0.5;
	if (prevClipPos.z > 0.99998) newClipPos.z = 0.9999985;
	if (all(greaterThan(newClipPos.xyz, vec3(0))) && all(lessThan(newClipPos.xyz, vec3(0.999999)))) {
		newClipPos.xy *= view;
		vec2 diff = newClipPos.xy - gl_FragCoord.xy + 0.01;
		ivec2 writePixelCoord = ivec2(gl_FragCoord.xy + floor(diff));
		uint depth = uint((1<<30) * newClipPos.z);
		imageAtomicMin(colorimg9, writePixelCoord, depth);
	}
	/*DRAWBUFFERS:3*/
}
#endif
//////Vertex Shader//////Vertex Shader//////
#ifdef VSH
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform int frameCounter;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

flat out mat4 localReprojectionMatrix;

void main() {
	vec3 dcamPos = previousCameraPosition - cameraPosition;
	localReprojectionMatrix =
		gbufferProjection *
		gbufferModelView *
		// the vec4s are interpreted as column vectors, not row vectors as suggested by this notation
		mat4(
			vec4(1, 0, 0, 0),
			vec4(0, 1, 0, 0),
			vec4(0, 0, 1, 0),
			vec4(dcamPos, 1)) * 
		gbufferPreviousModelViewInverse * 
		gbufferPreviousProjectionInverse;
	gl_Position = ftransform();
	reprojectionMatrix = localReprojectionMatrix;
}
#endif