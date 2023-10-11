#ifndef INCLUDE_VOLUMETRIC_BLOCKLIGHT
#define INCLUDE_VOLUMETRIC_BLOCKLIGHT

#define IRRADIANCECACHE_ONLY
#include "/lib/vx/SSBOs.glsl"

#ifndef LIGHTSHAFTS_ACTIVE
	float GetDepth(float depth) {
		return 2.0 * near * far / (far + near - (2.0 * depth - 1.0) * (far - near));
	}

	float GetDistX(float dist) {
		return (far * (dist - near)) / (dist * (far - near));
	}
#endif

vec3 GetVolumetricBlocklight(vec3 translucentMult, vec3 nViewPos, float z0, float z1, float dither) {
	if (max(blindness, darknessFactor) > 0.1) return vec3(0.0);

	vec3 volumetricBlocklight = vec3(0.0);

	float vlSceneIntensity = float(isEyeInWater == 1) * 0.5 + 0.5;
	#if LIGHTSHAFT_QUALI == 4
		int sampleCount = isEyeInWater == 1 ? 40 : 50;
	#elif LIGHTSHAFT_QUALI == 3
		int sampleCount = isEyeInWater == 1 ? 23 : 30;
	#elif LIGHTSHAFT_QUALI == 2
		int sampleCount = isEyeInWater == 1 ? 15 : 20;
	#elif LIGHTSHAFT_QUALI == 1
		int sampleCount = isEyeInWater == 1 ? 9 : 12;
	#endif

	float addition = 1.0;
	float maxDist = mix(max(far, 96.0) * 0.55, 80.0, vlSceneIntensity);

	#if WATER_FOG_MULT != 100
		if (isEyeInWater == 1) {
			#define WATER_FOG_MULT_M WATER_FOG_MULT * 0.01;
			maxDist /= WATER_FOG_MULT_M;
		}
	#endif

	float distMult = maxDist / (sampleCount + addition);
	float sampleMultIntense = isEyeInWater != 1 ? 1.0 : 0.85;

	float viewFactor = 1.0 - 0.7 * pow2(dot(nViewPos.xy, nViewPos.xy));

	float depth0 = GetDepth(z0);
	float depth1 = GetDepth(z1);

	// Fast but inaccurate perspective distortion approximation
	maxDist *= viewFactor;
	distMult *= viewFactor;
	
	float maxCurrentDist = min(depth1, maxDist);

	for (int i = 0; i < sampleCount; i++) {
		float currentDist = (i + dither) * distMult + addition;

		if (currentDist > maxCurrentDist) break;

		vec4 viewPos = gbufferProjectionInverse * (vec4(texCoord, GetDistX(currentDist), 1.0) * 2.0 - 1.0);
		viewPos /= viewPos.w;
		vec4 wpos = gbufferModelViewInverse * viewPos;
		vec3 playerPos = wpos.xyz / wpos.w;
		vec3 vxPos = playerPos + fract(cameraPosition);
		vec3 vlSample = vec3(0.0);

		float percentComplete = currentDist / maxDist;
		float sampleMult = mix(percentComplete * 3.0, sampleMultIntense, max(rainFactor, vlSceneIntensity));
		if (currentDist < 5.0) sampleMult *= smoothstep1(clamp(currentDist / 5.0, 0.0, 1.0));
		sampleMult /= sampleCount;
		if (isInRange(vxPos)) {
			vlSample = readVolumetricBlocklight(vxPos);
		}
		
		if (currentDist > depth0) vlSample *= translucentMult;

		volumetricBlocklight += vlSample * sampleMult;

	}
	
	volumetricBlocklight = max(volumetricBlocklight, vec3(0.0));

	#ifdef NETHER
		volumetricBlocklight *= VBL_NETHER_MULT;
	#elif defined END
		volumetricBlocklight *= VBL_END_MULT;
	#endif
	volumetricBlocklight *= VBL_STRENGTH;
	return volumetricBlocklight;
}
#endif
