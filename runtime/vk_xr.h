// VR through OpenXR (vk_xr.cpp), NX2NAVI_VR=1 at start. Off, the functions below do nothing and make no OpenXR call.
#pragma once

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

// vk_device.c create_instance: the OpenXR instance and the headset found, and the Vulkan instance extensions the runtime
// needs appended to ext[*n..cap)
void vk_xr_instance_exts(const char **ext, uint32_t *n, uint32_t cap);
// vk_device.c pick_device: the GPU the headset's runtime draws with (VK_NULL_HANDLE: no VR), and the one picked: VR goes
// off when they differ
VkPhysicalDevice vk_xr_device(VkInstance instance);
void vk_xr_device_picked(VkPhysicalDevice picked);
// vk_device.c create_device: the device extensions the runtime needs appended to ext[*n..cap), those the device has
void vk_xr_device_exts(const VkExtensionProperties *have, uint32_t n_have, const char **ext, uint32_t *n, uint32_t cap);
// vk_device.c create_device: the index of the graphics family's queue VR copies on, first_free when the family has it
// (the device is then made with one queue more), else UINT32_MAX and VR off
uint32_t vk_xr_queue(uint32_t family_queues, uint32_t first_free);
// vk_device.c, the device made: VR's queue and semaphore, and the session's thread started
void vk_xr_device_ready(VkInstance instance, VkPhysicalDevice phys, VkDevice device, uint32_t family);

// The head's view (NX2NAVI_VR_VIEW=head): the game's camera follows the headset. vr_game.c switches its camera hooks
// on when this is 1 at start
int vk_xr_head_wanted(void);
// The head's view is on now (the session running, the menu closed): vr_game.c's first person applies
int vk_xr_head_active(void);
// vr_game.c, the gameplay camera's update: 1 when the head's view applies now, the camera's position, target and up
// (game units) then turned and moved by the head (its yaw kept, the horizon level); the pose is kept for the picture
// drawn from it
int vk_xr_camera(double pos[3], double at[3], float up[3]);
// vr_game.c, the same update's angle load: 1 and the vertical angle (radians) the headset's lens needs, after a
// vk_xr_camera that applied
int vk_xr_lens(float *angle);
// vr_game.c: the game's own view this frame (an event's camera); its pictures go on the screen
void vk_xr_untracked(void);

// gfx_present.c present_image: 1 when VR takes this present's window image (then in TRANSFER_SRC_OPTIMAL for
// vk_xr_capture); 0 records nothing
int vk_xr_wants_capture(int hdr);
// The game's picture (picture, its layout and rect {x, y, w, h}) when it was drawn from the head, else the window's
// image, copied into one of VR's own images; *signal: what the present's submit signals for it
void vk_xr_capture(VkCommandBuffer cb, VkImage window, VkExtent2D extent, VkFormat format, VkImage picture,
                   VkImageLayout picture_layout, const int rect[4], VkSemaphoreSubmitInfo *signal);
// The present's submit went out with that signal: the image is the newest
void vk_xr_captured(const VkSemaphoreSubmitInfo *signal);

#ifdef __cplusplus
}
#endif
