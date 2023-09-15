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
uniform sampler2D colortex8;
//Program//
void main() {
    vec3 color = texelFetch(colortex3, texelCoord, 0).rgb;

	#ifdef FXAA
		FXAA311(color);
	#endif
	if (texCoord.x < 0.5) {
/*		vec4 playerPos = gbufferModelViewInverse * (gbufferProjectionInverse * vec4(texCoord * 2 - 1, 0.999, 1));
		playerPos.xyz = 40 * normalize(playerPos.xyz);
		ray_hit_t rayHit = raytrace(fract(cameraPosition), playerPos.xyz);
		color = (0.1 + 2 * float(rayHit.emissive)) * rayHit.rayColor.rgb * (dot(rayHit.normal, vec3(0.1, 0.3, 0.2)) + 0.8);
*/
		color = mix(min(texelFetch(colortex8, texelCoord, 0).rgb, vec3(1)), color, 0.5);
	}
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
