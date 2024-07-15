{
    for (int k = 0; k < 2; k++) {
        int index = int(gl_LocalInvocationIndex + k * 512);
        if (index > 1000) break;
        ivec3 localCoord = ivec3(index%10, index/10%10, index/100%10) - 1;
        ivec3 texCoord = localCoord + baseCoord;
        ivec3 prevTexCoord0 = texCoord + (1<<j) * floorCamPosOffset;
        ivec3 prevTexCoord = prevTexCoord0 + ivec3(0, (frameCounter % 2 * 2 + j/4) * voxelVolumeSize.y, 0);
        if ((thisOccupancy >> j & 1) == 1) {
            fullDist[localCoord.x+1][localCoord.y+1][localCoord.z+1] = (1.0-1.0/sqrt(3.0)) / (1<<j);
        } else if (
            all(greaterThanEqual(prevTexCoord0, ivec3(0))) &&
            all(lessThan(prevTexCoord0, voxelVolumeSize))
        ) {
            fullDist[localCoord.x+1][localCoord.y+1][localCoord.z+1] = imageLoad(distanceFieldI, prevTexCoord)[j%4] + 1.0 / (1<<j);
        } else {
            #if j > 0
                float prevDist = 10000.0;
                ivec3 prevCoord = prevTexCoord0 / 2 + voxelVolumeSize / 4 + ivec3(0, (frameCounter % 2 * 2 + (j-1)/4) * voxelVolumeSize.y, 0);
                ivec3 prevFractCoord = prevTexCoord0 % 2;
                for (int k = 0; k < 8; k++) {
                    ivec3 offset = ivec3(k%2, k/2%2, k/4%2) * (2 * prevFractCoord - 1);
                    vec3 distOffset = (0.75 - 0.5 * vec3(k == 0)) / (1<<(j-1));
                    prevDist = min(
                        prevDist,
                        min(min(distOffset.x, distOffset.y), distOffset.z)
                        + imageLoad(distanceFieldI, prevCoord + offset)[(j-1)%4]
                    );
                }
                fullDist[localCoord.x+1][localCoord.y+1][localCoord.z+1] = prevDist + 1.0/(1<<j);
            #else
                fullDist[localCoord.x+1][localCoord.y+1][localCoord.z+1] = 1000.0;
            #endif
        }
    }
    barrier();
    memoryBarrierShared();
    #if j > 0
        ivec3 prevTexCoord0 = texCoord + (1<<j) * floorCamPosOffset;
        ivec3 prevCoord = prevTexCoord0 / 2 + voxelVolumeSize / 4 + ivec3(0, (frameCounter % 2 * 2 + (j-1)/4) * voxelVolumeSize.y, 0);
        ivec3 prevFractCoord = prevTexCoord0 % 2;

        float prevDist = 10000.0;
        for (int k = 0; k < 8; k++) {
            ivec3 offset = ivec3(k%2, k/2%2, k/4%2) * (2 * prevFractCoord - 1);
            vec3 distOffset = (0.75 - 0.5 * vec3(k == 0)) / (1<<(j-1));
            prevDist = min(
                prevDist,
                min(min(distOffset.x, distOffset.y), distOffset.z)
                + imageLoad(distanceFieldI, prevCoord + offset)[(j-1)%4]
            );
        }
        if (prevDist < 3.0/(1<<j)) {
        #endif
        theseDists[j] = (thisOccupancy >> j & 1) == 1 ? -1.0/sqrt(3.0) / (1<<j) : ((thisOccupancy >> j+8 & 1) == 1 ? 0.5 : 1000);

        for (int k = 0; k < 27; k++) {
            ivec3 c2 = localCoord + ivec3(k%3, k/3%3, k/9%3);
            ivec3 c3 = localCoord + 2 * ivec3(k%3, k/3%3, k/9%3) - 1;
            theseDists[j] = min(theseDists[j], fullDist[c2.x][c2.y][c2.z]);
            theseDists[j] = min(theseDists[j], all(greaterThanEqual(c3, ivec3(0))) && all(lessThan(c3, ivec3(10))) ? fullDist[c3.x][c3.y][c3.z] + 1.0/(1<<j) : 1000);
        }
        #if j > 0
            if (prevDist > 2.0/(1<<j)) {
                theseDists[j] = mix(theseDists[j], prevDist,  prevDist * (1<<j) - 2.0);
            }
        } else {
            theseDists[j] = prevDist;
        }
        const float edgeDist = 4.0;
        float edgeDecider = min(infnorm(max(abs(texCoord + 0.5 - 0.5 * voxelVolumeSize) + edgeDist + 0.5 - 0.5 * voxelVolumeSize, 0.0)) / edgeDist, 1.0);
        if (edgeDecider > 0.01) {
            theseDists[j] = mix(theseDists[j], prevDist, edgeDecider);
        }
    #endif
    barrier();
}
