smoothnessG = pow2(color.g) * 1.5;
smoothnessG = min1(smoothnessG);
smoothnessD = smoothnessG;

/* Tweak to make caves with Glow Lichen look better lit and closer to vanilla Minecraft. */
lmCoordM = pow(lmCoordM + 0.0001, vec2(0.65));
