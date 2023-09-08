////////////////////////////////////////
// Complementary Reimagined by EminGT //
////////////////////////////////////////

//Common//
#include "/lib/common.glsl"

//////////Fragment Shader//////////Fragment Shader//////////Fragment Shader//////////
#ifdef FRAGMENT_SHADER

noperspective in vec2 texCoord;

//Uniforms//
uniform float viewWidth, viewHeight;

uniform sampler2D colortex3;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
//Pipeline Constants//
#ifndef TAA
	const bool colortex3MipmapEnabled = true;
#endif

//Common Variables//

//Common Functions//

//Includes//
#ifdef FXAA
	#include "/lib/antialiasing/fxaa.glsl"
#endif

#define DECLARE_CAMPOS
#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/raytrace.glsl"

//Program//
void main() {
    vec3 color = texelFetch(colortex3, texelCoord, 0).rgb;

	#ifdef FXAA
		FXAA311(color);
	#endif
	if (all(lessThan(gl_FragCoord.xy, vec2(400)))) {
		vec4 playerPos = gbufferModelViewInverse * (gbufferProjectionInverse * vec4(texCoord * 2 - 1, 0.999, 1));
		playerPos.xyz = 40 * normalize(playerPos.xyz);
		ray_hit_t rayHit = raytrace(fract(cameraPosition), playerPos.xyz);
		color = rayHit.rayColor.rgb * (dot(rayHit.normal, vec3(0.1, 0.3, 0.2)) + 0.8);
/*		color = vec3(fract(dot(gl_FragCoord.xy, vec2(0.1))) > 0.5);
		ivec3 coord = ivec3(gl_FragCoord.xy / 200 * (1<<(VOXEL_DETAIL_AMOUNT-1)), 0);
		voxel_t thisVoxel = readGeometry(getBaseIndex(6181, VOXEL_DETAIL_AMOUNT - 1), coord, VOXEL_DETAIL_AMOUNT-1);
		if (thisVoxel.color.a > 0.1) {
			color = thisVoxel.color.rgb;
		}
*/	}
    /*DRAWBUFFERS:3*/
	gl_FragData[0] = vec4(color, 1.0);
}

#endif

//////////Vertex Shader//////////Vertex Shader//////////Vertex Shader//////////
#ifdef VERTEX_SHADER

noperspective out vec2 texCoord;

//Uniforms//

//Attributes//

//Common Variables//

//Common Functions//

//Includes//

//Program//
void main() {
	gl_Position = ftransform();

	texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
}

#endif
