// VR's hooks in the game's camera update (Ocarina of Time 1.1.1), for vk_xr.cpp's head view (NX2NAVI_VR_VIEW=head).
//
// The camera update (main+0x4c48c0) chooses its position, target and up, then builds the look-at camera from them and,
// further on, the perspective projection from the camera's vertical angle. The free camera's site (debug_tools.c
// dbg_freecam_host, set freecam) sees the first in registers: for the gameplay camera the head's pose is put on them
// there (vr_camera_update). The set vrlens's site is the angle's load (vr_lens_host): the headset's lens replaces it
// for the same camera. Rendering and the game's culling both follow, as they read what the update made.
// While an event plays (a cutscene, a conversation) the events' camera follows the head instead, from where the event
// puts it. In first person, Link faces where the head looks (turn_link) and strafes when moved sideways or back (vr_pad).
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#include "cfg.h"
#include "debug_tools.h"
#include "host.h"
#include "layout.h"
#include "platform/platform.h"
#include "vk_xr.h"
#include "vr_wheel.h"
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

// First person's turn of Link toward the view while he is upright (not climbing, swimming or hanging), so what he holds
// and what reacts to his facing follow the head. His facing as the game keeps it, from which his rotation, his physics
// and the rest are made each frame: the object component 0 points at (+0x60), its 3x4 matrix by rows at +0x1f8 (the
// position in each row's fourth float). Its rotation is made the turn about the vertical toward the view. Running
// forward, the game turns him toward his movement, which the head's view steers; sideways or back he strafes (vr_pad).
static _Atomic uint64_t fp_at;   // when Link was last turned to the head in first person
static _Atomic uint64_t play_at;   // when the gameplay camera last updated outside an event (plat_now_ns)
static _Atomic int hooks_on;       // vr_game_start switched the camera hooks on: play_at is kept

// Link's state's name (his component 107, the name at its +0x244: "Wait", "CrouchGuard"...), or his upper body's
// (at +0x34c, "BowChargeUpper"...; with a weapon out, +0x31c "CommonUpper"), "" when there is none
static void link_state_at(char out[48], uint32_t at) {
    out[0] = 0;
    const uint64_t actor = dbg_player_actor();
    uint64_t components = 0, c107 = 0;
    if (!actor || !game_rw(actor + 0x270, 8)) return;
    memcpy(&components, (const void *)(uintptr_t)(actor + 0x270), 8);
    if (!game_rw(components + 107 * 8, 8)) return;
    memcpy(&c107, (const void *)(uintptr_t)(components + 107 * 8), 8);
    if (!game_rw(c107 + at, 47)) return;
    memcpy(out, (const void *)(uintptr_t)(c107 + at), 47);
    out[47] = 0;
}
static void link_state(char out[48]) { link_state_at(out, 0x244); }

// Link aims a ranged item: his upper body in the bow's (the slingshot's too) or the hookshot's states
static int link_aims(void) {
    char upper[48];
    link_state_at(upper, 0x34c);
    return !strncmp(upper, "Bow", 3) || !strncmp(upper, "Hookshot", 8);
}

// The game's thread at each pad read: whether Link plays the ocarina (a DemoOcarina state, a lesson's too), and VR's
// wheels. While one is open vk_xr.cpp holds the game's own quick select (L), which pauses the game. A wheel opens in play alone: the
// gameplay camera updated a moment ago (not the title, a load, an event), Link there and not in one of his scripted
// Demo states (the ocarina among them), and the game not paused by its menus (its three pause steps' flags). An item
// wheel's slot is equipped (its era's AssignableItem enum) a few reads after the release, once the quick select closed.
void vr_tick(Cpu *c) {
    char state[48];
    link_state(state);
    vk_xr_ocarina(!strncmp(state, "DemoOcarina", 11));   // playing the ocarina (a lesson's too)
    char weapon[48];   // his upper body with a weapon out: "CommonUpper" ("" once put away)
    link_state_at(weapon, 0x31c);
    vk_xr_weapon_out(weapon[0] && strcmp(weapon, "WeaponEquipOffUpper"));
    int slot;
    const int open = vk_xr_wheel(&slot) != 0;
    int p = 0, pre = 0, post = 0;
    dbg_flag_get("Sequence_IsPause", &p), dbg_flag_get("Sequence_IsPreGamePause", &pre);
    dbg_flag_get("Sequence_IsPostGamePause", &post);
    const int camera = !hooks_on || plat_now_ns() - play_at < 200000000ull;   // (without the hooks: Link's state alone)
    const int play = camera && !event_playing() && state[0] && strncmp(state, "Demo", 4);
    vk_xr_wheel_allow(open || (play && !(p || pre || post)));
    static int pending_slot = -1, pending_wait;
    int used_kind, used_slot;
    if (vk_xr_wheel_used(&used_kind, &used_slot) && used_kind == VR_WHEEL_ITEMS) pending_slot = used_slot, pending_wait = 12;
    if (pending_slot >= 0 && --pending_wait <= 0) {
        const char *item = vr_wheel_equip(pending_slot);
        const int child = vr_wheel_child();
        if (item) dbg_enum_set(c, child ? "PlayerStatus.AssignableItem.Child" : "PlayerStatus.AssignableItem.Adult", item);
        pending_slot = -1;
    }
}

// ov_input.c npad_states: in first person play, while the left stick is pushed sideways or back the game gets ZL held,
// so Link strafes as Z-targeting does, facing the way he faced (the head's, from running forward or standing). Pushed
// forward (within 45 degrees) he runs where the head looks, unless the head is within NX2NAVI_VR_RUN_ZONE degrees of
// its rest (the game's own camera's way, which the head turns from): then straight that way, so a head held about
// forward never steers him. The stick is turned so the game, which moves him relative to the head's view, runs him
// so, and turn_link faces him that way. The player's own ZL is kept.
static double head_s, head_c = 1;   // the head's view's level direction (sin, cos of its yaw), from the last camera update
static double base_s, base_c = 1;   // the game's own camera's, before the head turned it
static double run_s, run_c = 1;     // the way a run straight ahead takes
static uint64_t run_at;             // when the stick last ran him straight ahead (plat_now_ns)

void vr_pad(uint64_t *buttons, int32_t *lx, int32_t *ly) {
    if (plat_now_ns() - fp_at > 200000000ull) return;
    const int64_t m2 = (int64_t)*lx * *lx + (int64_t)*ly * *ly;
    if (m2 < (int64_t)8000 * 8000) return;
    if (!(*ly > 0 && 2 * (int64_t)*ly * *ly >= m2)) {
        *buttons |= 1u << 8;   // ZL
        return;
    }
    const double zone = cfg_int("NX2NAVI_VR_RUN_ZONE") * M_PI / 180.0;
    const double off = atan2(head_s * base_c - head_c * base_s, head_c * base_c + head_s * base_s);
    if (!(fabs(off) < zone)) return;
    // the stick's way from the rest's forward (s, c) and right (-c, s), then given relative to the head's
    const double wx = base_s * *ly - base_c * *lx, wz = base_c * *ly + base_s * *lx;
    const double f = wx * head_s + wz * head_c, r = -wx * head_c + wz * head_s;
    *ly = (int32_t)fmax(-32767, fmin(32767, f));
    *lx = (int32_t)fmax(-32767, fmin(32767, r));
    const double n = sqrt(wx * wx + wz * wz);
    run_s = wx / n, run_c = wz / n, run_at = plat_now_ns();
}

static void turn_link(const double ahead[3]) {
    const uint64_t actor = dbg_player_actor();
    if (!actor || strcmp(cfg_value("NX2NAVI_VR_CAMERA"), "first")) return;
    const double n = sqrt(ahead[0] * ahead[0] + ahead[2] * ahead[2]);
    if (!(n > 1e-4)) return;
    head_s = ahead[0] / n, head_c = ahead[2] / n;
    const int straight = plat_now_ns() - run_at < 100000000ull;   // running straight ahead: that way
    const float s = (float)(straight ? run_s : head_s), c = (float)(straight ? run_c : head_c);
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
    fp_at = plat_now_ns();
}

void vr_camera_update(Cpu *c, int scene) {
    static const int up_reg[3] = {7, 16, 6};
    if (scene != 1 && scene != 2) return;
    const int event = event_playing();
    const int k = scene - 1;
    if (scene == 1) head_camera = 0;
    if (scene == 2 ? !event : event) return;   // the camera that shows: Link's in play, the events' in an event
    if (scene == 1) play_at = plat_now_ns();
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
    if (scene == 1) {   // the game's own camera's level way: the head's rest, for a run straight ahead
        const double fx = at[0] - pos[0], fz = at[2] - pos[2], n = sqrt(fx * fx + fz * fz);
        if (n > 1e-6) base_s = fx / n, base_c = fz / n;
    }
    if (scene == 1) {   // aiming: the game's own camera's ray, where its shots go, for the head's view's reticle
        const int aims = link_aims();
        double d[3] = {at[0] - pos[0], at[1] - pos[1], at[2] - pos[2]};
        const double l = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        if (l > 1e-6) d[0] /= l, d[1] /= l, d[2] /= l;
        vk_xr_aim(aims && l > 1e-6, pos, d);
    }
    double eye[3];
    if (scene == 1 && vk_xr_head_active() && first_person_eye(eye))   // the game's camera's direction kept, from his eyes
        for (int i = 0; i < 3; i++) at[i] += eye[i] - pos[i], pos[i] = eye[i];
    if (scene == 2) {   // an event's camera raised by Link's height for his age, and NX2NAVI_VR_EVENT_UP cm more
        int child = 1;
        dbg_flag_get("Sequence_IsChildhood", &child);
        const double lift = (child ? 1.27 : 1.80) + cfg_int("NX2NAVI_VR_EVENT_UP") / 100.0;
        pos[1] += lift, at[1] += lift;
    }
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
    hooks_on = 1;
    rt_log("vr: the game's camera follows the headset (NX2NAVI_VR_VIEW=head)");
}
