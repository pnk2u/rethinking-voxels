const vec3[] LOGO_COLORS = vec3[](
    vec3(0.9, 0.23, 0.23),
    vec3(0.97, 0.41, 0.11),
    vec3(0.95, 0.48, 0.29),
    vec3(0.96, 0.75, 0.16),
    vec3(0.97, 0.71, 0.33),
    vec3(0.56, 0.85, 0.4),
    vec3(0.11, 0.73, 0.44),
    vec3(0.05, 0.68, 0.6),
    vec3(0.18, 0.87, 0.71),
    vec3(0.55, 0.96, 0.87),
    vec3(0.3, 0.39, 0.69),
    vec3(0.3, 0.6, 0.89),
    vec3(0.55, 0.82, 0.98),
    vec3(0.65, 0.51, 0.94),
    vec3(0.91, 0.67, 0.92),
    vec3(0.92, 0.49, 0.59)
);

const vec3[] CLOUD_POINTS = vec3[](
    vec3(0.2, 0.15, 0.05),
    vec3(0.6, 0.175, 0.075),
    vec3(0.37, 0.19, 0.09),
    vec3(0.28, 0.17, 0.07),
    vec3(0.5, 0.185, 0.085)
);

const vec3[] TREE_POINTS = vec3[](
    vec3(5.3, 0.8, 0.3),
    vec3(5.7, 1.0, 0.3),
    vec3(4.65, 1.1, 0.5),
    vec3(4.5, 1.5, 0.5),
    vec3(4.2, 1.3, 0.6),
    vec3(5.7, 1.4, 0.7),
    vec3(4.6, 1.85, 0.6),
    vec3(5.1, 1.9, 0.7)
);

const int flowerRowCount = 15;
const int petalCount = 16;

vec2 rand2(int u) {
    uint q = uint(u + 12347689);
    uvec2 n = q * uvec2(1597334673u, 3812015801u);
    n = (n.x ^ n.y) * uvec2(1597334673u, 3812015801u);
    return vec2(n) / float(0xffffffffu);
}

vec4 drawFlower(vec2 uv) {
    const float flowerHeight = 2.0;
    int index = int(uv.x * 4.0 + 1000.0) - 1000;
    uv = 4.0 * uv - vec2(float(index) + 0.5, -4.0);
    uv += (rand2(index) - 0.5) * 0.5;
    if (uv.y < 0.0 || uv.y > flowerHeight + 0.5) return vec4(0);
    vec2 flowerPetalUV = uv - vec2(0, flowerHeight);
    float flowerMidDist = length(flowerPetalUV);
    if (flowerMidDist < 0.13)
        return vec4(0.3, 0.3, 0.3, 1.0);
    if (flowerMidDist < 0.5) {
        for (int i = 0; i < petalCount; i++) {
            vec2 petalDir = vec2(cos(float(i) * 2.0 * pi / float(petalCount)), sin(float(i) * 2.0 * pi / float(petalCount)));
            float petalFactor = sqrt(1.0 - dot(petalDir, normalize(flowerPetalUV))) * 10.0/pi;
            if (petalFactor < 0.7 - 7.0 * pow2(flowerMidDist)) {
                float brightnessFactor = 1.0 - petalFactor * 0.2 + flowerMidDist * 0.2;
                vec3 color = LOGO_COLORS[(i + int(float(petalCount) * rand2(2 * index).x)) % petalCount];
                return vec4(brightnessFactor * color, 1);
            }
        }
    }
    if (uv.y > 0.0 && uv.y < flowerHeight && abs(uv.x - sin(3.0 * uv.y + 17.0 * cos(5.0 * float(index))) * 0.01) < 0.03) {
        float xfactor = (uv.x - sin(3.0 * uv.y + 17.0 * cos(5.0 * float(index))) * 0.01) / 0.03;
        return vec4(vec3(0.3, 0.5, 0.1) * (0.8 + 0.5 * sqrt(1.0 - xfactor * xfactor)), 1.0);
    }
    return vec4(0);
}

vec4 drawTree(vec2 uv) {
    int index = int(0.05 * uv.x + 100.0) - 100;
    uv.x = uv.x - 20.0 * float(index);
    uv += 0.3 * rand2(index);
    if (uv.x < 3.0 || uv.x > 7.0) return vec4(0);
    for (int i = 0; i < TREE_POINTS.length(); i++) {
        if (length(uv - TREE_POINTS[i].xy) < TREE_POINTS[i].z) {
            return vec4(vec3(0.2, 0.5, 0.1) * (1.0 + 0.2 * rand2(i + 10 * index).x), 1.0);
        }
    }
    if (uv.y < 0.8 && uv.y > -1.0 && abs(uv.x - 5.0 - 0.02 * uv.y) < 0.1 * (0.9 + 2.0 / pow2(uv.y + 2.0))) {
        return vec4(0.5, 0.3, 0.1, 1.0);
    }
    return vec4(0);
}

vec4 drawMountain(vec2 uv) {
    int mountainIndex = int(4.0 * uv.x + 100.0) - 100;
    vec2 thisHeightOffset = rand2(mountainIndex) - 0.5;
    vec2 nextHeightOffset = rand2(mountainIndex + 1) - 0.5;
    vec2 heightOffset = 0.1 * mix(thisHeightOffset, nextHeightOffset,
                        4.0 * uv.x - float(mountainIndex));
    int lowResMountainIndex = int(0.5 * uv.x + 100.0) - 100;
    vec2 lowResThisHeightOffset = rand2(lowResMountainIndex) - 0.5;
    vec2 lowResNextHeightOffset = rand2(lowResMountainIndex + 1) - 0.5;
    vec2 lowResHeightOffset = 0.7 * mix(lowResThisHeightOffset, lowResNextHeightOffset,
                        0.5 * uv.x - float(lowResMountainIndex));
    if (max(max((2.6 + lowResHeightOffset.y) * cos(0.15 * uv.x), cos(0.4 * uv.x + 1.3)), 0.0) - 1.0 > uv.y + heightOffset.x + heightOffset.y) {
        vec4 color =
            uv.y <-0.5 + 0.5 * heightOffset.x ? vec4(0.4, 0.7, 0.3, 1.0) : (
            uv.y < 0.6 + 0.7 * heightOffset.y ? vec4(0.3, 0.6, 0.4, 1.0) : (
            uv.y < 1.4 + 1.2 * heightOffset.x ? vec4(0.6, 0.6, 0.65, 1.0) : vec4(0.9, 0.9, 0.95, 1.0)));
            
        return color;
    }
    return vec4(0.0);
}

vec4 drawCloud(vec2 uv) {
    int index = int(0.1 * uv.x + 100.0) - 100;
    uv = (0.1 * uv - vec2(index, 0));
    if (uv.y < CLOUD_POINTS[0].y - CLOUD_POINTS[0].z) return vec4(0);
    if (uv.x < CLOUD_POINTS[0].x - CLOUD_POINTS[0].z) return vec4(0);
    if (uv.x > CLOUD_POINTS[1].x + CLOUD_POINTS[1].z) return vec4(0);
    for (int i = 0; i < CLOUD_POINTS.length(); i++) {
        if (length(uv - CLOUD_POINTS[i].xy) < CLOUD_POINTS[i].z) {
            return vec4(vec3(0.9, 0.9, 0.95) + 0.1 * float(uv.y > CLOUD_POINTS[0].y - 0.5 * CLOUD_POINTS[0].z), 1.0);
        }
    }
    if (uv.x > CLOUD_POINTS[0].x && uv.x < CLOUD_POINTS[1].x && uv.y > CLOUD_POINTS[0].y - CLOUD_POINTS[0].z && uv.y < CLOUD_POINTS[0].y) {
        return vec4(vec3(0.9, 0.9, 0.95) + 0.1 * float(uv.y > CLOUD_POINTS[0].y - 0.5 * CLOUD_POINTS[0].z), 1.0);
    }
    return vec4(0);
}

vec4 drawSun(vec2 uv) {
    vec2 relUV = uv - vec2(-0.7, 0.5);
    float dist = length(relUV);
    if (dist < 0.2) {
        return vec4(1.0, 0.95, 0.8, 1.0);
    }  
    float wind = -0.3 * frameTimeCounter;
    if (dist > 0.22 && dist < 0.3) {
        for (int i = 0; i < petalCount; i++) {
            vec2 petalDir = vec2(cos(float(i) * 2.0 * pi / float(petalCount) + wind), sin(float(i) * 2.0 * pi / float(petalCount) + wind));
            float petalFactor = sqrt(1.0 - dot(petalDir, normalize(relUV))) * 0.5 * float(petalCount) / pi;
            if (petalFactor < 0.4 - 4.0 * (dist - 0.2)) {
                float brightnessFactor = 1.0 - petalFactor * 0.2 + (dist - 0.2) * 0.2;
                vec3 color = LOGO_COLORS[i];
                return vec4(brightnessFactor * color, 1);
            }

        }
    }
    return vec4(0);
}

vec3 drawBackground() {
    vec4 fragColor = vec4(0);
    vec2 uv = gl_FragCoord.xy / viewHeight * 2.0 - 1.0;
    for (int k = 0; k < flowerRowCount; k++) {
        fragColor = drawFlower(uv * (1.0 + 0.1 * float(k)) + vec2(0.15 * frameTimeCounter + 24.0 * float(k), 0));
        if (fragColor.a > 0.5) {
            return fragColor.xyz;
        }
    }
    if ((1.0 + 0.1 * float(flowerRowCount - 1)) * uv.y < -1.0) {
        fragColor = vec4(vec3(0.8, 0.75, 0.5) * 0.7, 1.0);
        return fragColor.xyz;
    }
    fragColor = drawTree(uv * (1.0 + 0.1 * float(flowerRowCount + 2)) + 0.15 * vec2(frameTimeCounter, 0));
    if (fragColor.a > 0.5) return fragColor.xyz;
    const float cloudDist0 = 1.0 + 0.1 * float(flowerRowCount + 5);
    fragColor = drawCloud(uv * cloudDist0 + vec2(0.05 * frameTimeCounter, 0));
    if (fragColor.a > 0.5) return fragColor.xyz;
    const float mountainDist = 1.0 + 0.1 * float(flowerRowCount + 10);
    fragColor = drawMountain(mountainDist * uv + vec2(0.15 * frameTimeCounter, 0));
    if (fragColor.a > 0.5) return fragColor.xyz;
    const float cloudDist1 = 1.0 + 0.1 * float(flowerRowCount + 50);
    fragColor = drawCloud(uv * cloudDist1 + vec2(0.05 * frameTimeCounter, 0));
    if (fragColor.a > 0.5) return fragColor.xyz;
    fragColor = drawSun(uv);
    if (fragColor.a > 0.5) return fragColor.xyz;
    return vec3(0.5, 0.7, 1.0);
}