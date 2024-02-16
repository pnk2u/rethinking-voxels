#ifndef SSBO
    #define SSBO
    #ifndef WRITE_TO_SSBOS
        #define READONLY
        #define WRITE_TO_SSBOS readonly
    #endif

    struct light_t {
        int mat;
        ivec3 blockPos;
        ivec4 subPos;
        ivec4 col;
    };

    layout(std430, binding=0) WRITE_TO_SSBOS buffer stuff {
        mat4 gbufferPreviousModelViewInverse;
        mat4 gbufferPreviousProjectionInverse;
        mat4 reprojectionMatrix;
        int globalLightCount;
        light_t[] globalLightList;
    };
    #ifdef READONLY
        #undef WRITE_TO_SSBOS
    #endif
#endif
