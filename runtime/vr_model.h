// A model of the game's files for VR's hands (vr_model.c): an actor's mesh and its colour texture, read from the
// user's RomFS. Nothing is shipped.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The mesh's vertices by 8 floats (position xyz in metres, normal xyz, texture uv), its triangles' indices, and the
// colour texture (RGBA8, sRGB-encoded), all malloc'd
typedef struct {
    float *verts;
    uint32_t *index;
    uint32_t verts_n, index_n;
    uint8_t *rgba;
    uint32_t tex_w, tex_h;
} VrModel;

// Model/<actor>.<actor>.bfres.zs's shapes at their finest detail and the texture its "_Alb" names; 1 when read
int vr_model_load(const char *actor, VrModel *out);
void vr_model_free(VrModel *m);

#ifdef __cplusplus
}
#endif
