#include "/lib/common.glsl"
#ifdef FRAGMENT_SHADER

#include "/lib/vx/voxelReading.glsl"

void main() {
    ivec3 floorCamPos = cameraPositionInt;
    ivec3 camOffset = cameraPositionInt - previousCameraPositionInt;
    if (cameraPositionInt.y == -98257195) {
        floorCamPos = ivec3(floor(cameraPosition) + 0.5 * sign(cameraPosition));
        camOffset = ivec3(1.1 * (floorCamPos - floor(previousCameraPosition)));
    }
    int waveResolution = min(8, int(shadowMapResolution / far));

    vec3 pos0 = vec3((gl_FragCoord.xy - 0.5 * shadowMapResolution) / waveResolution, 30).xzy;
    int waterHeight = 63 - floorCamPos.y;
    bool hasEntity = false;
    if (all(greaterThan(pos0.xz, -0.5 * voxelVolumeSize.xz)) && all(lessThan(pos0.xz, 0.5 * voxelVolumeSize.xz))) {
        waterHeight = -1000;
        for (int k = 0; k < 60; k++) {
            ivec3 coords = ivec3(pos0.xz + voxelVolumeSize.xz/2, 30 - k + voxelVolumeSize.y/2).xzy;
            int waterData = imageLoad(voxelCols, coords * ivec3(1, 2, 1)).r >> 26;
            if ((waterData & 1) == 1) {
                waterHeight = coords.y - voxelVolumeSize.y/2;
                pos0.y = waterHeight + 0.5;
                hasEntity = (waterData & 2) == 2;
                break;
            }
        }
    }
    ivec2 prevCoords = ivec2(gl_FragCoord.xy) + camOffset.xz * waveResolution;
    vec4 data = texelFetch(shadowcolor2, prevCoords, 0);

    if (data.z < -999.0) {
        data.xy *= 0.9;
    }
    #define LOC data.x
    #define SPEED data.y
    #define EXTRADATA data.z

    const float wavespeed = 2.4 * waveResolution;
    float dt = min(0.5 / wavespeed, frameTime);

    if (waterHeight >= -500) {
        EXTRADATA = waterHeight + 0.5;
        mat4 aroundData = mat4(
            texelFetchOffset(shadowcolor2, prevCoords, 0, ivec2( 1, 0)),
            texelFetchOffset(shadowcolor2, prevCoords, 0, ivec2(-1, 0)),
            texelFetchOffset(shadowcolor2, prevCoords, 0, ivec2(0, 1)),
            texelFetchOffset(shadowcolor2, prevCoords, 0, ivec2(0,-1))
        );
        ivec4 aroundHeights = ivec4(floor(transpose(aroundData)[2]) + 1000.5) - 1000 - camOffset.y;
        for (int i = 0; i < 4; i++) {
            if (abs(aroundHeights[i] - waterHeight) >= 2) {
                aroundData[i].x = LOC;
            }
        }
        float laplaceloc
            = aroundData[0].x
            + aroundData[1].x
            + aroundData[2].x
            + aroundData[3].x
            - 4 * LOC;
        SPEED += laplaceloc * wavespeed * wavespeed * dt;
        LOC += SPEED * dt;
        LOC *= 0.996;
        SPEED *= 0.996;
    } else {
        data.zw = vec2(-999.5);
    }

    if (hasEntity) {
        data.y += dt * sin(frameTimeCounter * 2) * max(0.0, 1.0 - 2 * length(fract(pos0.xz) - 0.5));
    }
    vec3 playerPos = pos0 - relativeEyePosition - cameraPositionFract;
    float playerLen = length(playerPos);
    if (playerLen < 1.5) {
        data.x -= 0.1 * (1.5 - playerLen) * dot(playerPos, cameraPosition - previousCameraPosition);
    }
    if (frameCounter < 100) {
        data = vec4(0);
    }
    #undef LOC
    #undef SPEED
    #undef EXTRADATA
    /* RENDERTARGETS:2 */
    gl_FragData[0] = data;
}

#endif

#ifdef VERTEX_SHADER
void main() {
    #ifdef INTERACTIVE_WATER
        gl_Position = ftransform();
    #else
        gl_Position = vec4(-1);
    #endif
}
#endif