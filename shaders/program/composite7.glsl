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

layout(std430, binding=0) readonly buffer blockidmap {
    int blockIdMap[];
};

//Program//
void main() {
	
	if (gl_FragCoord.x < 512 && gl_FragCoord.y < 20) {
		int coord = int(gl_FragCoord.x) + 512 * int(gl_FragCoord.y);
		int mappedMat = blockIdMap[coord];
		gl_FragData[0] = vec4(
			mappedMat % 256,
			mappedMat / 256,
			0, 255) / 255.0;
		return;
	}
    vec3 color = texelFetch(colortex3, texelCoord, 0).rgb;

	#ifdef FXAA
		FXAA311(color);
	#endif

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
