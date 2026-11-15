// VR's hooks in the game's camera update (Ocarina of Time 1.1.1), for vk_xr.cpp's head view (NX2NAVI_VR_VIEW=head).
//
// The camera update (main+0x4c48c0) chooses its position, target and up, then builds the look-at camera from them and,
// further on, the perspective projection from the camera's vertical angle. The free camera's site (debug_tools.c
// dbg_freecam_host, set freecam) sees the first in registers: for the gameplay camera the head's pose is put on them
// there (vr_camera_update). The set vrlens's site is the angle's load (vr_lens_host): the headset's lens replaces it
// for the same camera. Rendering and the game's culling both follow, as they read what the update made.
// While an event plays (a cutscene, a conversation) the events' camera follows the head instead, from where the event
// puts it. In first person, Link faces where the head looks while he stands (turn_link).
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "cfg.h"
#include "debug_tools.h"
#include "host.h"
#include "layout.h"
#include "vk_xr.h"
#include "vr_game.h"

static uint64_t head_camera;   // x19 of the update the head's view was put on (0: none); the game's thread alone

static double reg_d(const Cpu *c, int r) {
    uint64_t b = (uint64_t)c->v[r];
    double v;
    memcpy(&v, &b, 8);
    return v;
}
static void set_d(Cpu *c, int r, double v) {
    uint64_t b;
    memcpy(&b, &v, 8);
    c->v[r] = b;
}
static float reg_s(const Cpu *c, int r) {
    uint32_t b = (uint32_t)c->v[r];
    float v;
    memcpy(&v, &b, 4);
    return v;
}
static void set_s(Cpu *c, int r, float v) {
    uint32_t b;
    memcpy(&b, &v, 4);
    c->v[r] = b;
}

// An event plays (a cutscene, a conversation): the event manager's singleton (main+0x4c7e090) holds the manager,
// whose +104 is the event running, 0 when none. The event camera updates every frame whether one plays or not.
static int event_playing(void) {
    uint64_t singleton = 0, events = 0, running = 0;
    memcpy(&singleton, (const void *)(uintptr_t)(MOD_MAIN_BASE + 0x4c7e090), 8);
    if (singleton) memcpy(&events, (const void *)(uintptr_t)singleton, 8);
    if (events) memcpy(&running, (const void *)(uintptr_t)(events + 104), 8);
    return running != 0;
}

// n bytes at a are the game's read-write memory (a pointer read from the game, before it is followed)
static int game_rw(uint64_t a, size_t n) {
    RtMemoryInfo mi;
    return a && !mem_query(a, &mi) && mi.state && (mi.perm & 3) == 3 && a + n <= mi.base + mi.size;
}

// First person's hiding of Link: the game fades him out when its camera comes close, through two of his components'
// alphas (the components at actor +0x270; at their +0x260 an object of vtable main+0x4b4a5a0 with its alpha at +0x118,
// at +0x278 one of vtable main+0x4b4b4a8 with its alpha at +0x130). Below zero they stay hidden through the game's
// easing back toward 1, every frame; what he holds is drawn by other actors and stays. Written while first person is on.
static void hide_link(uint64_t actor) {
    static const struct { uint32_t slot; uint64_t vtable; uint32_t field; } fades[] = {
        {0x260, MOD_MAIN_BASE + 0x4b4a5a0, 0x118}, {0x278, MOD_MAIN_BASE + 0x4b4b4a8, 0x130}};
    static const float hidden = -1.0f;
    uint64_t components = 0;
    memcpy(&components, (const void *)(uintptr_t)(actor + 0x270), 8);
    if (!game_rw(components, 0x280)) return;
    for (unsigned i = 0; i < 2; i++) {
        uint64_t object = 0, vtable = 0;
        memcpy(&object, (const void *)(uintptr_t)(components + fades[i].slot), 8);
        if (!game_rw(object, fades[i].field + 4)) continue;
        memcpy(&vtable, (const void *)(uintptr_t)object, 8);
        if (vtable == fades[i].vtable) dbg_poke(object + fades[i].field, &hidden, sizeof hidden);
    }
}

// First person (NX2NAVI_VR_CAMERA=first): the camera at Link's eyes, from his feet, his facing and his age; the
// offsets NX2NAVI_VR_EYE_UP, _SIDE and _FORWARD (centimetres, -30..30) move it from the place found right in a Quest 3
// (child 1.27 m, adult 1.80 m over his feet, 8 cm ahead of them). 0 while there is no Link.
static int first_person_eye(double eye[3]) {
    if (strcmp(cfg_value("NX2NAVI_VR_CAMERA"), "first")) return 0;
    const uint64_t actor = dbg_player_actor();
    float feet[3], m[9];
    if (!actor || !dbg_player_pos(feet) || !dbg_player_rot(m)) return 0;
    double fx = m[2], fz = m[8], n = sqrt(fx * fx + fz * fz);   // his facing: the rotation's third column, level
    if (!(n > 1e-4)) return 0;
    fx /= n, fz /= n;
    int child = 1;
    dbg_flag_get("Sequence_IsChildhood", &child);
    const double up = (child ? 1.27 : 1.80) + cfg_int("NX2NAVI_VR_EYE_UP") / 100.0;
    const double ahead = 0.08 + cfg_int("NX2NAVI_VR_EYE_FORWARD") / 100.0, side = cfg_int("NX2NAVI_VR_EYE_SIDE") / 100.0;
    eye[0] = feet[0] + ahead * fx - side * fz;   // his right: (-fz, 0, fx)
    eye[1] = feet[1] + up;
    eye[2] = feet[2] + ahead * fz + side * fx;
    hide_link(actor);
    return 1;
}

// First person's turn of Link toward the view, while he stands upright (not climbing, swimming
// or hanging), so what he holds and what reacts to his facing follow the head. His facing as the game keeps it, from
// which his rotation, his physics and the rest are made each frame: the object component 0 points at (+0x60), its 3x4
// matrix by rows at +0x1f8 (the position in each row's fourth float). Its rotation is made the turn about the vertical
// toward the view. Walking, the game turns him toward his movement, which the head's view already steers.
static void turn_link(const double ahead[3]) {
    static float last[3];
    static int have_last;
    float feet[3];
    const uint64_t actor = dbg_player_actor();
    if (!actor || !dbg_player_pos(feet)) {
        have_last = 0;
        return;
    }
    const float dx = feet[0] - last[0], dz = feet[2] - last[2];
    const int standing = have_last && dx * dx + dz * dz < 0.004f * 0.004f;
    memcpy(last, feet, sizeof last), have_last = 1;
    if (!standing || strcmp(cfg_value("NX2NAVI_VR_CAMERA"), "first")) return;
    const double n = sqrt(ahead[0] * ahead[0] + ahead[2] * ahead[2]);
    if (!(n > 1e-4)) return;
    const float s = (float)(ahead[0] / n), c = (float)(ahead[2] / n);
    uint64_t components = 0, c0 = 0, facing = 0;
    memcpy(&components, (const void *)(uintptr_t)(actor + 0x270), 8);
    if (game_rw(components, 8)) memcpy(&c0, (const void *)(uintptr_t)components, 8);
    if (game_rw(c0, 0x68)) memcpy(&facing, (const void *)(uintptr_t)(c0 + 0x60), 8);
    float t[12];
    if (!game_rw(facing, 0x1f8 + sizeof t)) return;
    memcpy(t, (const void *)(uintptr_t)(facing + 0x1f8), sizeof t);
    if (fabsf(t[5] - 1.0f) > 0.01f) return;   // not upright now
    const float r[12] = {c, 0, s, t[3], 0, 1, 0, t[7], -s, 0, c, t[11]};
    dbg_poke(facing + 0x1f8, r, sizeof r);
}

void vr_camera_update(Cpu *c, int scene) {
    static const int up_reg[3] = {7, 16, 6};
    if (scene != 1 && scene != 2) return;
    const int event = event_playing();
    const int k = scene - 1;
    if (scene == 1) head_camera = 0;
    if (scene == 2 ? !event : event) return;   // the camera that shows: Link's in play, the events' in an event
    double pos[3], at[3];
    float up[3];
    for (int i = 0; i < 3; i++) pos[i] = reg_d(c, i), at[i] = reg_d(c, 3 + i), up[i] = reg_s(c, up_reg[i]);
    // the view put on last time handed back as it was (the game did not choose its camera again): the game's own from
    // before it is used, so the head is never put on twice
    static double in_pos[2][3], in_at[2][3], out_pos[2][3], out_at[2][3];
    static float in_up[2][3];
    static int have_out[2];
    static unsigned fed_back;
    if (have_out[k] && !memcmp(pos, out_pos[k], sizeof pos) && !memcmp(at, out_at[k], sizeof at)) {
        memcpy(pos, in_pos[k], sizeof pos), memcpy(at, in_at[k], sizeof at), memcpy(up, in_up[k], sizeof up);
        if (!fed_back++) rt_log("vr: the game's camera handed back the head's view: its own view from before used");
    }
    memcpy(in_pos[k], pos, sizeof pos), memcpy(in_at[k], at, sizeof at), memcpy(in_up[k], up, sizeof up);
    have_out[k] = 0;
    double eye[3];
    if (scene == 1 && vk_xr_head_active() && first_person_eye(eye))   // the game's camera's direction kept, from his eyes
        for (int i = 0; i < 3; i++) at[i] += eye[i] - pos[i], pos[i] = eye[i];
    if (!vk_xr_camera(pos, at, up)) return;
    for (int i = 0; i < 3; i++) set_d(c, i, pos[i]), set_d(c, 3 + i, at[i]), set_s(c, up_reg[i], up[i]);
    memcpy(out_pos[k], pos, sizeof pos), memcpy(out_at[k], at, sizeof at), have_out[k] = 1;
    head_camera = c->x[19];
    if (scene != 1) return;   // Link turns in play alone
    const double ahead[3] = {at[0] - pos[0], at[1] - pos[1], at[2] - pos[2]};
    turn_link(ahead);
}

void vr_lens_host(Cpu *c) {
    float angle;
    memcpy(&angle, (const void *)(uintptr_t)(c->x[19] + 0x13c), sizeof angle);   // the instruction: ldr s0, [x19, #0x13c]
    float headset;
    if (head_camera && c->x[19] == head_camera && vk_xr_lens(&headset) && isfinite(headset) && headset > 0.05f &&
        headset < 3.0f)
        angle = headset;
    set_s(c, 0, angle);
}

void vr_game_start(void) {
    const int lens = rt_patch_set("vrlens");
    if (lens < 0 || rt_patch_set("freecam") < 0) {
        rt_log("vr: this game version has no camera sites for the head's view: the picture stays on the screen");
        return;
    }
    rt_patch_switch(lens, 1);
    dbg_camera_watch(1);   // the free camera's site, where vr_camera_update runs
    rt_log("vr: the game's camera follows the headset (NX2NAVI_VR_VIEW=head)");
}
