#include "/lib/common.glsl"

#ifdef CSH_A
const vec2 workGroupsRender = vec2(1.0, 1.0);
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

uniform sampler2D colortex10;
layout(rgba16f) uniform writeonly image2D colorimg12;

shared vec3 readColors[10][10];
shared vec4 normalDepthDatas[10][10];


void main() {
    ivec2 texelCoord = ivec2(gl_GlobalInvocationID);
    #if BLOCKLIGHT_RESOLUTION == 1
        imageStore(
            colorimg12,
            texelCoord,
            texelFetch(colortex10, texelCoord, 0)
        );
    #else
        int index = int(gl_LocalInvocationID.x) + 16 * int(gl_LocalInvocationID.y);
        ivec2 lowerBound = ivec2(gl_WorkGroupID.xy) * 16 / BLOCKLIGHT_RESOLUTION;
        ivec2 upperBound = ivec2(gl_WorkGroupID.xy + 1) * 16 / BLOCKLIGHT_RESOLUTION + 2;
        ivec2 readSize = upperBound - lowerBound;
        if (index < readSize.x * readSize.y) {
            ivec2 readCoords = ivec2(index % readSize.x, index / readSize.x);
            readColors[readCoords.x][readCoords.y] = texelFetch(colortex10, readCoords + lowerBound, 0).rgb;
            vec4 normalDepthData = texelFetch(colortex8, BLOCKLIGHT_RESOLUTION * (readCoords + lowerBound), 0);
            normalDepthData.w *= 100.0;
            normalDepthDatas[readCoords.x][readCoords.y] = normalDepthData;
        }
        barrier();
        memoryBarrierShared();
        vec3 writeColor = vec3(0);
        float totalWeight = 1e-5;
        vec2 lrTexelCoord = vec2(texelCoord) / BLOCKLIGHT_RESOLUTION;
        ivec2 localLrTexCoord = ivec2(lrTexelCoord) - lowerBound;
        vec2 weights = fract(lrTexelCoord);//smoothstep(vec2(0.0), vec2(1.0), fract(lrTexelCoord));
        vec4 normalDepthData = texelFetch(colortex8, texelCoord, 0);
        normalDepthData.w *= 100.0;
        for (int k = 0; k < 4; k++) {
            ivec2 valueOffset = ivec2(k%2, k/2);
            ivec2 c = localLrTexCoord + valueOffset;
            float weight =
                max(1e-5, 1.0 - 5.0 * length(normalDepthData - normalDepthDatas[c.x][c.y])) *
                mix(1.0 - weights.x, weights.x, valueOffset.x) *
                mix(1.0 - weights.y, weights.y, valueOffset.y);
            writeColor += readColors[c.x][c.y] * weight;
            totalWeight += weight;
        }
        imageStore(colorimg12, texelCoord, vec4(writeColor / totalWeight, 1.0));
    #endif
}
#endif