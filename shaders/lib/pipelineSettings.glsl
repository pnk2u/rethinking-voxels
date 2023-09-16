/*
const int colortex0Format = R11F_G11F_B10F;	//main color
const int colortex1Format = RGB8;			//smoothnessD & materialMask & skyLightFactor
const int colortex2Format = RGBA16;			//taa, inverted depth
const int colortex3Format = RGB8;			//*cloud map on deferred* & translucentMult & bloom & final color
const int colortex4Format = RGBA8;			//volumetric cloud linear depth & volumetric light factor & normalM in composite
const int colortex5Format = RGBA8_SNORM;	//normalM & scene image for water reflections
#ifdef TEMPORAL_FILTER
const int colortex6Format = R16;			//previous depth
const int colortex7Format = RGBA16F;		//*cloud map on gbuffers* & temporal filter
#endif
const int colortex8Format = RGBA16F;		//reprojected normal and depth data in prepare
const int colortex9Format = RGBA32UI;		//scaled depth for atomics in reprojection validation
const int colortex10Format= RGBA16F;		//block lighting
*/

const bool colortex0Clear = true;
const bool colortex1Clear = true;
const bool colortex2Clear = false;
const bool colortex3Clear = true;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
#ifdef TEMPORAL_FILTER
const bool colortex6Clear = false;
const bool colortex7Clear = false;
#endif

const int noiseTextureResolution = 128;

const bool shadowHardwareFiltering = true;
const float shadowDistanceRenderMul = 1.0;
const float entityShadowDistanceMul = 0.125; // Iris feature

const float drynessHalflife = 300.0;
const float wetnessHalflife = 300.0;

const float ambientOcclusionLevel = 1.0;