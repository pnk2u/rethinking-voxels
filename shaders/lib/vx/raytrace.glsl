#ifndef RAYTRACE
#define RAYTRACE
#include "/lib/vx/voxelMapping.glsl"
#include "/lib/vx/voxelReading.glsl"

const mat3 eye = mat3(
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
);
// cuboid intersection algorithm
float aabbIntersect(vxData data, vec3 pos, vec3 dir, inout int n) {
    // offset to work around floating point errors
    vec3 offset = 0.001 * eye[n] * sign(dir[n]);
    // don't need to know global position, only relative to current block
    pos = fract(pos + offset) - offset;
    vec3[2] bounds = vec3[2](data.lower, data.upper);
    float w = 10000;
    for (int i = 0; i < 3; i++) {
        if (dir[i] == 0) continue;
        float relevantBound = bounds[dir[i] < 0 ? 1 : 0][i];
        float w0 = (relevantBound - pos[i]) / dir[i];
        if (w0 < -0.00005 / length(dir)) {
            relevantBound = bounds[dir[i] < 0 ? 0 : 1][i];
            w0 = (relevantBound - pos[i]) / dir[i];
        }
        vec3 newPos = pos + w0 * dir;
        // ray-plane intersection position needs to be closer than the previous best one and further than approximately 0
        bool valid = (w0 > -0.00005 / length(dir) && w0 < w);
        for (int j = 1; j < 3; j++) {
            int ij = (i + j) % 3;
            // intersection position also needs to be within other bounds
            if (newPos[ij] < bounds[0][ij] || newPos[ij] > bounds[1][ij]) {
                valid = false;
                break;
            }
        }
        // update normal and ray position
        if (valid) {
            w = w0;
            n = i;
        }
    }
    return w;
}
// returns color data of the block at pos, when hit by ray in direction dir
vec4 handledata(inout vxData data, sampler2D atlas, inout vec3 pos, vec3 dir, int n) {
    vec4 color = vec4(0);
    vec3 offset = 0.001 * eye[n] * sign(dir[n]);
    if (!data.crossmodel) {
        bool hit = true;
        if (data.cuboid) {
            float w = aabbIntersect(data, pos, dir, n);
            if (w > 9999) hit = false;
            else pos += w * dir;
        }
        if (hit) {
            vec2 spritecoord = vec2(n != 0 ? fract(pos.x) : fract(pos.z), n != 1 ? fract(-pos.y) : fract(pos.z)) * 2 - 1;
            ivec2 texcoord = ivec2(data.texcoord * atlasSize + (data.spritesize - 0.5) * spritecoord);
            color = texelFetch(atlas, texcoord, 0);
            if (!data.alphatest) color.a = 1;
            // multiply by vertex color for foliage, water etc
            color.rgb *= data.emissive ? vec3(1) : data.lightcol;
        }
    } else {
        // get around floating point errors using an offset
        vec3 blockInnerPos = fract(pos + offset) - offset;
        // ray-plane intersections
        float w0 = (1 - blockInnerPos.x - blockInnerPos.z) / (dir.x + dir.z);
        float w1 = (blockInnerPos.x - blockInnerPos.z) / (dir.z - dir.x);
        vec3 p0 = blockInnerPos + w0 * dir;
        vec3 p1 = blockInnerPos + w1 * dir;
        bool valid0 = (max(max(abs(p0.x - 0.5), 0.8 * abs(p0.y - 0.5)), abs(p0.z - 0.5)) < 0.4);
        bool valid1 = (max(max(abs(p1.x - 0.5), 0.8 * abs(p1.y - 0.5)), abs(p1.z - 0.5)) < 0.4);
        vec4 color0 = valid0 ? texelFetch(atlas, ivec2(data.texcoord * atlasSize + (data.spritesize - 0.5) * (1 - p0.xy * 2)), 0) : vec4(0);
        vec4 color1 = valid1 ? texelFetch(atlas, ivec2(data.texcoord * atlasSize + (data.spritesize - 0.5) * (1 - p1.xy * 2)), 0) : vec4(0);
        color0.xyz *= data.emissive ? vec3(1) : data.lightcol;
        color1.xyz *= data.emissive ? vec3(1) : data.lightcol;
        pos += (valid0 ? w0 : (valid1 ? w1 : 0)) * dir;
        // the more distant intersection position only contributes by the amount of light coming through the closer one
        color = (w0 < w1) ? (vec4(color0.xyz * color0.a, color0.a) + (1 - color0.a) * vec4(color1.xyz * color1.a, color1.a)) : (vec4(color1.xyz * color1.a, color1.a) + (1 - color1.a) * vec4(color0.xyz * color0.a, color0.a));
    }
    if ((data.cuboid || data.crossmodel) && data.mat != 31000 && color.a < 0.1) {
        float w = (0.875 - fract(pos.y + offset.y)) / dir.y;
        if (w > 0) pos += w * dir;
        float aroundWater = 0;
        for (int k = 0; k < 4; k++) {
            vec3 pos1 = pos + offset + (2 * ((k >> 1) % 2) == 0 ? vec3(2 * (k % 2) - 1, 0, 0) : vec3(0, 0, 2 * (k % 2) - 1));
            vxData aroundVxData = readVxMap(getVxPixelCoords(pos1));
            if (isInRange(pos1) && aroundVxData.mat == 31000) aroundWater += aroundVxData.upper.y;
        }
        if (aroundWater < 1.5) return color;
        int colData = int(texelFetch(colortex10, ivec2(pos.xz + vxRange / 2.0), 0).w * 65535 + 0.5);
        vec3 waterCol = vec3(((colData >> 8) % 8) / 7.0, ((colData >> 11) % 8) / 7.0, ((colData >> 14) % 4) / 3.5);
        data.mat = 31000;
        return vec4(waterCol, 0.5);
    }
    return color;
}
// voxel ray tracer
vec4 raytrace(bool lowDetail, inout vec3 pos0, bool doScattering, vec3 dir, inout vec3 translucentHit, sampler2D atlas, bool translucentData) {
    vec3 progress;
    for (int i = 0; i < 3; i++) {
        //set starting position in each direction
        progress[i] = -(dir[i] < 0 ? fract(pos0[i]) : fract(pos0[i]) - 1) / dir[i];
    }
    int i = 0;
    // get closest starting position
    float w = progress[0];
    for (int i0 = 1; i0 < 3; i0++) {
        if (progress[i0] < w) {
            i = i0;
            w = progress[i];
        }
    }
    // step size in each direction (to keep to the voxel grid)
    vec3 stp = abs(1 / dir);
    float dirlen = length(dir);
    float invDirLenScaled = 0.001 / dirlen;
    vec3 dirsgn = sign(dir);
    vec3[3] eyeOffsets;
    for (int k = 0; k < 3; k++) {
        eyeOffsets[k] = 0.0001 * eye[k] * dirsgn[k];
    }
    vec3 pos = pos0 + invDirLenScaled * dir;
    vec3 scatterPos = pos0;
    vec4 raycolor = vec4(0);
    vec4 oldRayColor = vec4(0);
    const float scatteringMaxAlpha = 0.1;
    // check if stuff already needs to be done at starting position
    vxData voxeldata = readVxMap(getVxPixelCoords(pos));
    bool isScattering = false;
    if (lowDetail && voxeldata.full && !voxeldata.alphatest) return vec4(0, 0, 0, translucentData ? 0 : 1);
    if (isInRange(pos) && voxeldata.trace && !lowDetail) {
        raycolor = handledata(voxeldata, atlas, pos, dir, i);
        if (doScattering && raycolor.a > 0.1) isScattering = (voxeldata.mat == 10004 || voxeldata.mat == 10008 || voxeldata.mat == 10016);
        if (doScattering && isScattering) {
            scatterPos = pos;
            raycolor.a = min(scatteringMaxAlpha, raycolor.a);
        }
        raycolor.rgb *= raycolor.a;
    }
    if (raycolor.a > 0.01 && raycolor.a < 0.9) translucentHit = pos;
    int k = 0; // k is a safety iterator
    int mat = raycolor.a > 0.1 ? voxeldata.mat : 0; // for inner face culling
    vec3 oldPos = pos;
    bool oldFull = voxeldata.full;
    // main loop
    while (w < 1 && k < 2000 && raycolor.a < 0.99) {
        oldRayColor = raycolor;
        pos = pos0 + (min(w, 1.0)) * dir + eyeOffsets[i];
        // read voxel data at new position and update ray colour accordingly
        if (isInRange(pos)) {
            voxeldata = readVxMap(getVxPixelCoords(pos));
            pos -= eyeOffsets[i];
            if (lowDetail) {
                if (voxeldata.trace && voxeldata.full && !voxeldata.alphatest) {
                    pos0 = pos + eyeOffsets[i];
                    return vec4(0, 0, 0, translucentData ? 0 : 1);
                }
            } else {
                bool newScattering = false;
                if (voxeldata.trace) {
                    vec4 newcolor = handledata(voxeldata, atlas, pos, dir, i);
                    if (dot(pos - pos0, dir) < 0.0) newcolor.a = 0;
                    bool samemat = voxeldata.mat == mat;
                    mat = (newcolor.a > 0.1) ? voxeldata.mat : 0;
                    if (doScattering) newScattering = (mat == 10004 || mat == 10008 || mat == 10016);
                    if (newScattering) newcolor.a = min(newcolor.a, scatteringMaxAlpha);
                    if (samemat) newcolor.a = clamp(10.0 * newcolor.a - 9.0, 0.0, 1.0);
                    raycolor.rgb += (1 - raycolor.a) * newcolor.a * newcolor.rgb;
                    raycolor.a += (1 - raycolor.a) * newcolor.a;
                    if (oldRayColor.a < 0.01 && raycolor.a > 0.01 && raycolor.a < 0.9) translucentHit = pos;
                }
                if (doScattering) {
                    if (isScattering) {
                        scatterPos = pos;
                    }
                    oldFull = voxeldata.full;
                    oldPos = pos;
                    isScattering = newScattering;
                }
            }
            #ifdef CAVE_SUNLIGHT_FIX
            if (!isInRange(pos, 2)) {
                int height = int(texelFetch(colortex10, ivec2(pos.xz + floor(cameraPosition.xz) - floor(previousCameraPosition.xz) + vxRange / 2), 0).w * 65535 + 0.5) % 256 - VXHEIGHT * VXHEIGHT / 2;
                if (pos.y + floor(cameraPosition.y) - floor(previousCameraPosition.y) < height) {
                    raycolor.a = 1;
                }
            }
            #endif
            pos += eyeOffsets[i];
        }
        // update position
        k += 1;
        progress[i] += stp[i];
        w = progress[0];
        i = 0;
        for (int i0 = 1; i0 < 3; i0++) {
            if (progress[i0] < w) {
                i = i0;
                w = progress[i];
            }
        }
    }
    float oldAlpha = raycolor.a;
    raycolor.a = 1 - exp(-4*length(scatterPos - pos0)) * (1 - raycolor.a);
    raycolor.rgb += raycolor.a - oldAlpha; 
    pos0 = pos;
    raycolor = (k == 2000 ? vec4(1, 0, 0, 1) : raycolor);
    return translucentData ? oldRayColor : raycolor;
}
vec4 raytrace(bool lowDetail, inout vec3 pos0, vec3 dir, inout vec3 translucentHit, sampler2D atlas, bool translucentData) {
    return raytrace(lowDetail, pos0, false, dir, translucentHit, atlas, translucentData);
}
vec4 raytrace(inout vec3 pos0, bool doScattering, vec3 dir, sampler2D atlas) {
    vec3 translucentHit = vec3(0);
    return raytrace(false, pos0, doScattering, dir, translucentHit, atlas, false);
}
vec4 raytrace(inout vec3 pos0, vec3 dir, inout vec3 translucentHit, sampler2D atlas, bool translucentData) {
    return raytrace(false, pos0, dir, translucentHit, atlas, translucentData);
}
vec4 raytrace(bool lowDetail, inout vec3 pos0, vec3 dir, sampler2D atlas) {
    vec3 translucentHit = vec3(0);
    return raytrace(lowDetail, pos0, dir, translucentHit, atlas, false);
}
vec4 raytrace(bool lowDetail, inout vec3 pos0, vec3 dir, sampler2D atlas, bool translucentData) {
    vec3 translucentHit = vec3(0);
    return raytrace(lowDetail, pos0, dir, translucentHit, atlas, translucentData);
}
vec4 raytrace(inout vec3 pos0, vec3 dir, sampler2D atlas, bool translucentData) {
    vec3 translucentHit = vec3(0);
    return raytrace(pos0, dir, translucentHit, atlas, translucentData);
}
vec4 raytrace(inout vec3 pos0, vec3 dir, sampler2D atlas) {
    return raytrace(pos0, dir, atlas, false);
}
#endif