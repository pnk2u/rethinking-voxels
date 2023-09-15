#include "/lib/common.glsl"

//////Fragment Shader//////Fragment Shader//////
#ifdef FSH
uniform sampler2D colortex8;

const ivec2 offsets[8] = ivec2[8](
	ivec2( 1, 0),
	ivec2( 1, 1),
	ivec2( 0, 1),
	ivec2(-1, 1),
	ivec2(-1, 0),
	ivec2(-1,-1),
	ivec2( 0,-1),
	ivec2( 1,-1));

void main() {
	ivec2 texelCoord = ivec2(gl_FragCoord.xy);
	vec4 writeData = texelFetch(colortex8, texelCoord, 0);
	if (writeData.a > 1.5) {
		vec4 avgAroundData = vec4(0);
		int validAroundCount = 0;
		bool extendable = true;
		for (int k = 0, invalidInARow = 0; k < 10; k++) {
			vec4 aroundData = texelFetch(colortex8, texelCoord + offsets[k%8], 0);
			if (aroundData.a > 1.5) {
				invalidInARow++;
				if (invalidInARow >= 4) {
					extendable = false;
					break;
				}
				continue;
			}
			invalidInARow = 0;
			if (k < 8) {
				avgAroundData += aroundData;
				validAroundCount++;
			}
		}
		if (extendable) {
			writeData = avgAroundData / validAroundCount;
		}
	}
	/*RENDERTARGETS:8*/
	gl_FragData[0] = writeData;
}
#endif

//////Vertex Shader//////Vertex Shader//////
#ifdef VSH

void main() {
	gl_Position = ftransform();
}
#endif
