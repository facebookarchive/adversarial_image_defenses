#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void generatePatches(
        float* result, // N x (C x P x P)
        float* img, // C x H x W
        unsigned int imgH,
        unsigned int imgW,
        unsigned int patchSize,
        unsigned int overlap);

void generateQuiltedImages(
        float* result, // C x H x W
        long* neighbors, // M
        float* patchDict, // N x (C x P x P)
        unsigned int imgH,
        unsigned int imgW,
        unsigned int patchSize,
        unsigned int overlap,
        bool graphcut);

#ifdef __cplusplus
}
#endif
