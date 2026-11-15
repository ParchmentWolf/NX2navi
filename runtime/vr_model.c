// A model of the game's files for VR's hands (vr_model.h).
//
// BFRES (version 10): the header's +0x28 points at the models, the buffer section's offset is at the memory pool info
// the header's +0xb0 points at (+8). A model (FMDL) points at its vertex sets (+0x20) and shapes (+0x28) and counts
// them (+0x68, +0x6a). A vertex set (FVTX, 0x60 bytes) points at its attributes (+0x08: 0x10 each, the name's string,
// the format, the offset in the vertex) and strides (+0x38), keeps its data's offset in the buffer section (+0x48), its
// attribute count (+0x4c) and its vertex count (+0x50). A shape (FSHP, 0x60 bytes) names its vertex set (+0x10) and
// meshes (+0x18), its levels of detail, the finest first: a mesh (0x38 bytes) has its indices' offset in the buffer
// section (+0x20), primitive (+0x24, 3 triangles), index size (+0x28, 1 16 bits, 2 32 bits), count (+0x2c) and first
// vertex (+0x30). Strings are a 16-bit length and the bytes. Textures are apart, Tex/<name>.bntx.zs.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "host.h"
#include "menu_assets.h"
#include "vr_model.h"

static uint16_t le16(const uint8_t *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t le32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24; }
static uint64_t le64(const uint8_t *p) { return (uint64_t)le32(p) | (uint64_t)le32(p + 4) << 32; }

static float half(uint16_t h) {
    const int e = h >> 10 & 31, m = h & 1023;
    const float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? INFINITY : ldexpf((float)(m | 1024), e - 25);
    return h & 0x8000 ? -v : v;
}

// an attribute's first three (or two) components as floats; 0 for a format not known
static int attr(const uint8_t *p, uint32_t fmt, float out[3]) {
    switch (fmt) {
    case 0x1505:   // 16_16_16_16 float
    case 0x1205:   // 16_16 float
        for (int i = 0; i < (fmt == 0x1505 ? 3 : 2); i++) out[i] = half(le16(p + 2 * i));
        return 1;
    case 0x1805:   // 32_32_32 float
    case 0x1d05:   // 32_32_32_32 float
        memcpy(out, p, 12);
        return 1;
    case 0x1201:   // 16_16 unorm
        out[0] = le16(p) / 65535.0f, out[1] = le16(p + 2) / 65535.0f;
        return 1;
    case 0x1202:   // 16_16 snorm
        out[0] = (int16_t)le16(p) / 32767.0f, out[1] = (int16_t)le16(p + 2) / 32767.0f;
        return 1;
    case 0x0e02: {   // 10_10_10_2 snorm
        const uint32_t v = le32(p);
        for (int i = 0; i < 3; i++) {
            int c = (int)(v >> (10 * i) & 1023);
            if (c & 512) c -= 1024;
            out[i] = fmaxf(-1.0f, c / 511.0f);
        }
        return 1;
    }
    default:
        return 0;
    }
}

// the first string of the file ending "_Alb" (a material's colour texture), into name
static int albedo_name(const uint8_t *b, size_t n, char *name, size_t cap) {
    for (size_t i = 0; i + 6 < n; i++) {
        const uint16_t len = le16(b + i);
        if (len < 5 || len >= cap || i + 2 + len >= n || b[i + 2 + len] != 0) continue;
        const uint8_t *s = b + i + 2;
        if (memcmp(s + len - 4, "_Alb", 4)) continue;
        int ok = 1;
        for (uint16_t k = 0; k < len && ok; k++) ok = s[k] > 32 && s[k] < 127;
        if (!ok) continue;
        memcpy(name, s, len), name[len] = 0;
        return 1;
    }
    return 0;
}

void vr_model_free(VrModel *m) {
    free(m->verts), free(m->index), free(m->rgba);
    memset(m, 0, sizeof *m);
}

int vr_model_load(const char *actor, VrModel *out) {
    memset(out, 0, sizeof *out);
    char path[160];
    snprintf(path, sizeof path, "Model/%s.%s.bfres.zs", actor, actor);
    size_t n = 0;
    uint8_t *b = menu_asset_load(path, &n);
    if (!b || n < 0x100 || memcmp(b, "FRES", 4)) {
        rt_log("vr: hands: %s not read", path);
        free(b);
        return 0;
    }
#define IN(o, k) ((uint64_t)(o) + (uint64_t)(k) <= n)
    int ok = 0;
    const uint64_t pool = le64(b + 0xb0), model = le64(b + 0x28);
    if (!IN(pool, 16) || !IN(model, 0x70) || memcmp(b + model, "FMDL", 4)) goto done;
    const uint64_t buffers = le64(b + pool + 8), shapes = le64(b + model + 0x28);
    const uint16_t shape_n = le16(b + model + 0x6a);
    // two passes over the shapes: their sizes, then their data
    for (int pass = 0; pass < 2; pass++) {
        uint32_t nv = 0, ni = 0;
        for (uint16_t si = 0; si < shape_n; si++) {
            const uint64_t s = shapes + 0x60ull * si;
            if (!IN(s, 0x60) || memcmp(b + s, "FSHP", 4)) goto done;
            const uint64_t vt = le64(b + s + 0x10), mesh = le64(b + s + 0x18);
            if (!IN(vt, 0x60) || !IN(mesh, 0x38)) goto done;
            const uint32_t count = le32(b + vt + 0x50), stride = IN(le64(b + vt + 0x38), 4) ? le32(b + le64(b + vt + 0x38)) : 0;
            const uint64_t data = buffers + le32(b + vt + 0x48);
            const uint32_t prim = le32(b + mesh + 0x24), isize = le32(b + mesh + 0x28) == 2 ? 4 : 2;
            const uint32_t icount = le32(b + mesh + 0x2c), first = le32(b + mesh + 0x30);
            const uint64_t ib = buffers + le32(b + mesh + 0x20);
            if (prim != 3 || !stride || !IN(data, (uint64_t)count * stride) || !IN(ib, (uint64_t)icount * isize)) continue;
            if (pass == 1) {
                uint32_t pos_fmt = 0, nrm_fmt = 0, uv_fmt = 0, pos_at = 0, nrm_at = 0, uv_at = 0;
                const uint64_t attrs = le64(b + vt + 0x08);
                for (uint8_t a = 0; a < b[vt + 0x4c]; a++) {
                    const uint64_t at = attrs + 0x10ull * a;
                    if (!IN(at, 0x10)) break;
                    const uint64_t no = le64(b + at);
                    if (!IN(no, 5)) continue;
                    const char *nm = (const char *)b + no + 2;
                    const uint32_t f = le32(b + at + 8), off = le16(b + at + 12);
                    if (!memcmp(nm, "_p0", 3)) pos_fmt = f, pos_at = off;
                    else if (!memcmp(nm, "_n0", 3)) nrm_fmt = f, nrm_at = off;
                    else if (!memcmp(nm, "_u0", 3)) uv_fmt = f, uv_at = off;
                }
                for (uint32_t v = 0; v < count; v++) {
                    const uint8_t *src = b + data + (uint64_t)v * stride;
                    float *d = out->verts + 8ull * (nv + v), p[3] = {0}, q[3] = {0, 1, 0}, u[3] = {0};
                    if (!attr(src + pos_at, pos_fmt, p)) {
                        rt_log("vr: hands: %s has positions of format %#x", path, pos_fmt);
                        goto done;
                    }
                    if (nrm_fmt) attr(src + nrm_at, nrm_fmt, q);
                    if (uv_fmt) attr(src + uv_at, uv_fmt, u);
                    d[0] = p[0], d[1] = p[1], d[2] = p[2], d[3] = q[0], d[4] = q[1], d[5] = q[2], d[6] = u[0], d[7] = u[1];
                }
                for (uint32_t i = 0; i < icount; i++) {
                    const uint32_t k = isize == 4 ? le32(b + ib + 4ull * i) : le16(b + ib + 2ull * i);
                    out->index[ni + i] = nv + first + k < nv + count ? nv + first + k : nv;
                }
            }
            nv += count, ni += icount - icount % 3;
        }
        if (pass == 0) {
            if (!nv || !ni) goto done;
            out->verts = malloc(sizeof(float) * 8 * nv), out->index = malloc(sizeof(uint32_t) * ni);
            if (!out->verts || !out->index) goto done;
        } else {
            out->verts_n = nv, out->index_n = ni;
        }
    }
    char tex[96];
    if (albedo_name(b, n, tex, sizeof tex)) {
        char tpath[160];
        snprintf(tpath, sizeof tpath, "Tex/%s.bntx.zs", tex);
        size_t tn = 0;
        uint8_t *t = menu_asset_load(tpath, &tn);
        if (!t || !menu_asset_texture(t, tn, tex, &out->rgba, &out->tex_w, &out->tex_h))
            rt_log("vr: hands: %s's texture %s not read", actor, tpath);
        free(t);
    }
    if (!out->rgba) {   // a plain grey then
        out->rgba = malloc(4), out->tex_w = out->tex_h = 1;
        if (out->rgba) memset(out->rgba, 160, 4);
    }
    ok = out->rgba != NULL;
    if (ok) rt_log("vr: hands: %s: %u vertices, %u triangles, texture %ux%u", actor, out->verts_n, out->index_n / 3, out->tex_w, out->tex_h);
done:
#undef IN
    free(b);
    if (!ok) vr_model_free(out);
    return ok;
}
