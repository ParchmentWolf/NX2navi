// VR through OpenXR (vk_xr.h), NX2NAVI_VR=1 at start. Stage one, the theater: the picture the window shows, the menu
// over it, on a screen floating in front of the headset (NX2NAVI_VR_SCREEN=<width m>,<distance m>, 3.2,2.5).
//
// The loader is opened at run time (libopenxr_loader.so.1, openxr_loader.dll), so a PC without it starts as before.
// XR_KHR_vulkan_enable: the runtime names the Vulkan instance and device extensions it needs and the GPU it draws with;
// vk_device.c adds the extensions, prefers that GPU and makes the device with one more queue of the graphics family for
// VR's copies, so the session's thread never waits on the game's queue lock.
//
// Each present copies the window's image into one of three images of VR's own (the boxes, vk_xr_capture), and the
// present's submit signals a timeline semaphore with the copy's number. The session's thread copies the newest box into
// the OpenXR swapchain on VR's queue, its submit waiting for that number, and shows it as a quad in the LOCAL space. A
// box the thread reads is never written, and neither is the newest, so three are enough; the headset's frames go on at
// its own rate, the last picture shown again until a newer one is there.
#define XR_NO_PROTOTYPES
#define XR_USE_GRAPHICS_API_VULKAN
#include <vulkan/vulkan.h>

#include "third_party/openxr/include/openxr/openxr.h"
#include "third_party/openxr/include/openxr/openxr_platform.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

extern "C" {
#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "ui/menu.h"
#include "vk_xr.h"
#include "vr_game.h"
#include "xr_depth_keep_spv.h"
#include "xr_depth_spv.h"
#include "xr_hud_spv.h"
int vk_dlaa_vr_planes(float planes[2]);                     // vk_dlaa.h
void rec_drain(void);                                        // vk_rec.h
}

namespace {

#define XR_FUNCS(X)                                                                                                    \
    X(xrDestroyInstance) X(xrGetInstanceProperties) X(xrGetSystem) X(xrGetSystemProperties) X(xrPollEvent)            \
    X(xrCreateSession) X(xrBeginSession) X(xrEndSession) X(xrWaitFrame) X(xrBeginFrame) X(xrEndFrame)                 \
    X(xrCreateReferenceSpace) X(xrEnumerateSwapchainFormats) X(xrCreateSwapchain) X(xrEnumerateSwapchainImages)       \
    X(xrAcquireSwapchainImage) X(xrWaitSwapchainImage) X(xrReleaseSwapchainImage) X(xrGetVulkanInstanceExtensionsKHR) \
    X(xrGetVulkanDeviceExtensionsKHR) X(xrGetVulkanGraphicsDeviceKHR) X(xrGetVulkanGraphicsRequirementsKHR)          \
    X(xrLocateViews) X(xrLocateSpace)

// Tangents of a lens's four edges (left and down negative), as OpenXR's angles are
struct Lens {
    float l = -1, r = 1, u = 1, d = -1;
};
struct Pose {
    float q[4] = {0, 0, 0, 1};   // x, y, z, w
    float p[3] = {0, 0, 0};
};
// What a picture was drawn from: the head's pose in the LOCAL space and the lens; tracked false: the game's own view,
// shown on the screen
struct Drawn {
    bool tracked = false;
    bool hud = false;      // the HUD apart: the box holds the scene without it, the HUD box its panel
    bool stereo = false;   // depth stereo: the box holds the left eye's picture, the right box the right eye's
    bool menu = false;     // the game paused (its menus): the boxes hold the world as it last was, the HUD box the
                           // game's picture, its menu, shown on the HUD's panel
    Pose pose;             // the head's, between the eyes
    Lens lens;
    float scale = 1.0f;    // the lens the game drew with over the one shown (NX2NAVI_VR_VIEW_SCALE)
    float ipd = 0.064f;    // the eyes' distance, metres
};

struct Xr {
    bool tried = false, ok = false;
    PFN_xrGetInstanceProcAddr gipa = nullptr;
#define XR_DECLARE(f) PFN_##f f = nullptr;
    XR_FUNCS(XR_DECLARE)
#undef XR_DECLARE
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId system = XR_NULL_SYSTEM_ID;
    std::string instance_exts, device_exts;   // the runtime's lists, split in place: the pointers below point into them
    std::vector<const char *> instance_list, device_list;
    float screen_width = 3.2f, screen_distance = 2.5f;
    // the device (vk_xr_device_ready)
    VkInstance vk_instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t family = 0, queue_index = UINT32_MAX;
    VkQueue queue = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    // the swapchain's size (the window's, from the first capture) and order; ready once the session's thread made it
    std::atomic<uint32_t> want_w{0}, want_h{0};
    std::atomic<bool> ready{false}, rgba{true};
    uint32_t width = 0, height = 0;
    // the boxes (mu): the present's thread writes one, the session's thread reads the newest
    std::mutex mu;
    VkImage boxes[3]{}, boxes_r[3]{};   // (the right eye's, depth stereo)
    uint64_t box_value[3]{};
    int latest = -1, reading = -1, next_slot = 0, pending = -1;
    bool boxes_made = false, boxes_failed = false;
    uint64_t next_value = 0;   // the present's thread alone
    Drawn box_drawn[3];
    // the head (the session's thread writes, the game's threads read)
    std::mutex head_mu;
    bool have_head = false;
    Pose head;
    Lens eyes;   // both eyes' lenses together
    float ipd = 0.064f;
    std::atomic<bool> running{false};
    // the view now (each present): the head's on the game's camera, or the screen (NX2NAVI_VR_VIEW, the menu open)
    std::atomic<bool> head_view{false};
    std::atomic<float> picture_ratio{16.0f / 9.0f};
    // this frame's (frame_mu): taken at the gameplay camera's update (vk_xr_camera), its lens given at the angle's
    // load (vk_xr_lens), kept for the next picture (vk_xr_capture)
    std::mutex frame_mu;
    bool frame_on = false;
    Drawn drawn;
    float render_scale = 1.0f;   // the lens the game draws with over the one shown (NX2NAVI_VR_VIEW_SCALE)
    std::atomic<uint32_t> camera_given{0}, lens_given{0}, tracked_pictures{0};
    // the gameplay camera's updates since the last picture, and the pictures in a row with none: a paused game (the
    // inventory) draws flat pictures while the camera stands still, and those go on the screen
    std::atomic<uint32_t> updates{0};
    int stale = 0;   // the present's thread alone
    // the HUD's panel (NX2NAVI_VR_HUD=panel). The game draws its HUD over its upscaled scene in one float target; the
    // scene as the upscaler left it is kept (vk_xr_scene_image, the replay's stream), and at the present, when the
    // target holds the HUD too, shaders/xr_hud.comp makes the scene without the HUD and the HUD's panel from the two
    std::mutex hud_mu;
    VkImage hud_target = VK_NULL_HANDLE, scene = VK_NULL_HANDLE, scene8 = VK_NULL_HANDLE, panel8 = VK_NULL_HANDLE;
    VkImageView target_view = VK_NULL_HANDLE, scene_view = VK_NULL_HANDLE, scene8_view = VK_NULL_HANDLE, panel8_view = VK_NULL_HANDLE;
    VkImage target_viewed = VK_NULL_HANDLE;   // the target target_view is of
    uint32_t hud_w = 0, hud_h = 0;
    VkFormat hud_format = VK_FORMAT_UNDEFINED;
    uint64_t kept = 0, used = 0;   // scenes kept, and the last one a panel was made from
    VkDescriptorSetLayout hud_set = VK_NULL_HANDLE;
    VkPipelineLayout hud_layout = VK_NULL_HANDLE;
    VkPipeline hud_pipe = VK_NULL_HANDLE;
    VkSampler nearest = VK_NULL_HANDLE;
    PFN_vkCmdPushDescriptorSetKHR push_set = nullptr;
    bool hud_tried = false;
    VkImage hud_boxes[3]{};
    std::atomic<uint32_t> panels{0};
    std::atomic<uint32_t> pic_w{0}, pic_h{0};   // the game's presented picture's size: the HUD's target is the scene upscaled to it
    // depth stereo (NX2NAVI_VR_DEPTH, hud_mu): shaders/xr_depth.comp makes each eye's picture from the scene the HUD's
    // shader leaves and the scene's depth kept by vk_xr_keep_depth
    VkImage eye8[2]{};
    VkImageView eye8_view[2]{};
    uint32_t eye8_w = 0, eye8_h = 0;
    // the scene's depth as it was when the scene was drawn (vk_xr_keep_depth): the game reuses its target for smaller
    // passes before the present
    VkImage depth = VK_NULL_HANDLE;   // R32F
    VkImageView depth_view = VK_NULL_HANDLE;
    uint32_t depth_w = 0, depth_h = 0;
    uint64_t depth_kept = 0;
    VkDescriptorSetLayout keep_set = VK_NULL_HANDLE;
    VkPipelineLayout keep_layout = VK_NULL_HANDLE;
    VkPipeline keep_pipe = VK_NULL_HANDLE;
    VkSampler keep_sampler = VK_NULL_HANDLE;
    PFN_vkCmdPushDescriptorSetKHR keep_push = nullptr;
    bool keep_tried = false;
    VkDescriptorSetLayout depth_set = VK_NULL_HANDLE;
    VkPipelineLayout depth_layout = VK_NULL_HANDLE;
    VkPipeline depth_pipe = VK_NULL_HANDLE;
    VkSampler linear = VK_NULL_HANDLE;
    bool depth_tried = false;
    std::atomic<uint32_t> stereo_pictures{0}, stereo_missed{0}, menu_pictures{0};
    // the last picture drawn from the head whose world scene8 and eye8 still hold (the present's thread alone): a paused
    // game's menus go on the HUD's panel in front of that world, frozen as it was
    Drawn frozen;
};

Xr &xr() {
    static Xr x;
    return x;
}

// "a b c" into its names, the string's spaces made ends
void split(std::string &s, std::vector<const char *> &out) {
    out.clear();
    for (char &ch : s)
        if (ch == ' ') ch = 0;
    for (size_t i = 0; i < s.size();) {
        if (s[i]) out.push_back(s.c_str() + i), i += std::strlen(s.c_str() + i);
        else i++;
    }
}

bool init() {
    Xr &x = xr();
    if (x.tried) return x.ok;
    x.tried = true;
    if (!cfg_bool("NX2NAVI_VR")) return false;
    if (const char *e = cfg_get("NX2NAVI_VR_SCREEN")) {
        float w = 0, d = 0;
        if (std::sscanf(e, "%f,%f", &w, &d) == 2 && w > 0.1f && d > 0.1f) x.screen_width = w, x.screen_distance = d;
    }
#ifdef _WIN32
    void *lib = plat_lib_open("openxr_loader.dll");
#else
    // WiVRn installed from Flathub registers no runtime for programs outside the sandbox: its manifest is used when no
    // runtime is named
    char named[8];
    if (!plat_getenv("XR_RUNTIME_JSON", named, sizeof named)) {
        static const char *wivrn = "/var/lib/flatpak/app/io.github.wivrn.wivrn/current/active/files/share/openxr/1/openxr_wivrn.json";
        if (plat_file_exists(wivrn)) {
            plat_setenv("XR_RUNTIME_JSON", wivrn);
            rt_log("vr: XR_RUNTIME_JSON unset: WiVRn's Flatpak runtime, %s", wivrn);
        }
    }
    void *lib = plat_lib_open("libopenxr_loader.so.1");
#endif
    if (lib) x.gipa = (PFN_xrGetInstanceProcAddr)plat_lib_sym(lib, "xrGetInstanceProcAddr");
    if (!x.gipa) {
        rt_log("vr: no OpenXR loader: VR off");
        return false;
    }
    PFN_xrCreateInstance create = nullptr;
    PFN_xrEnumerateInstanceExtensionProperties enumerate = nullptr;
    x.gipa(XR_NULL_HANDLE, "xrCreateInstance", (PFN_xrVoidFunction *)&create);
    x.gipa(XR_NULL_HANDLE, "xrEnumerateInstanceExtensionProperties", (PFN_xrVoidFunction *)&enumerate);
    if (!create || !enumerate) {
        rt_log("vr: the OpenXR loader lacks xrCreateInstance: VR off");
        return false;
    }
    uint32_t n = 0;
    enumerate(nullptr, 0, &n, nullptr);
    std::vector<XrExtensionProperties> exts(n, {XR_TYPE_EXTENSION_PROPERTIES});
    enumerate(nullptr, n, &n, exts.data());
    bool vulkan = false;
    for (const auto &e : exts) vulkan |= !std::strcmp(e.extensionName, XR_KHR_VULKAN_ENABLE_EXTENSION_NAME);
    if (!vulkan) {
        rt_log("vr: the OpenXR runtime has no %s (is its server running?): VR off", XR_KHR_VULKAN_ENABLE_EXTENSION_NAME);
        return false;
    }
    const char *want[] = {XR_KHR_VULKAN_ENABLE_EXTENSION_NAME};
    XrInstanceCreateInfo ci{XR_TYPE_INSTANCE_CREATE_INFO};
    std::strcpy(ci.applicationInfo.applicationName, "NX2navi");
    std::strcpy(ci.applicationInfo.engineName, "NX2navi");
    ci.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
    ci.enabledExtensionCount = 1;
    ci.enabledExtensionNames = want;
    XrResult r = create(&ci, &x.instance);
    if (XR_FAILED(r)) {
        rt_log("vr: xrCreateInstance failed (%d; is the runtime's server running and the headset connected?): VR off", (int)r);
        return false;
    }
#define XR_LOAD(f) x.gipa(x.instance, #f, (PFN_xrVoidFunction *)&x.f);
    XR_FUNCS(XR_LOAD)
#undef XR_LOAD
    XrInstanceProperties ip{XR_TYPE_INSTANCE_PROPERTIES};
    x.xrGetInstanceProperties(x.instance, &ip);
    XrSystemGetInfo sg{XR_TYPE_SYSTEM_GET_INFO};
    sg.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    r = x.xrGetSystem(x.instance, &sg, &x.system);
    if (XR_FAILED(r)) {
        rt_log("vr: %s %u.%u.%u: no headset (%d): VR off", ip.runtimeName, XR_VERSION_MAJOR(ip.runtimeVersion),
               XR_VERSION_MINOR(ip.runtimeVersion), XR_VERSION_PATCH(ip.runtimeVersion), (int)r);
        x.xrDestroyInstance(x.instance), x.instance = XR_NULL_HANDLE;
        return false;
    }
    XrSystemProperties sp{XR_TYPE_SYSTEM_PROPERTIES};
    x.xrGetSystemProperties(x.instance, x.system, &sp);
    XrGraphicsRequirementsVulkanKHR req{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};   // asked before a session, as the spec says
    x.xrGetVulkanGraphicsRequirementsKHR(x.instance, x.system, &req);
    for (int pass = 0; pass < 2; pass++) {
        auto get = pass ? x.xrGetVulkanDeviceExtensionsKHR : x.xrGetVulkanInstanceExtensionsKHR;
        std::string &s = pass ? x.device_exts : x.instance_exts;
        uint32_t len = 0;
        get(x.instance, x.system, 0, &len, nullptr);
        s.assign(len, '\0');
        if (len) get(x.instance, x.system, len, &len, s.data());
        while (!s.empty() && !s.back()) s.pop_back();
        split(s, pass ? x.device_list : x.instance_list);
    }
    rt_log("vr: %s %u.%u.%u, %s; Vulkan %u.%u to %u.%u; instance extensions \"%s\", device extensions \"%s\"",
           ip.runtimeName, XR_VERSION_MAJOR(ip.runtimeVersion), XR_VERSION_MINOR(ip.runtimeVersion),
           XR_VERSION_PATCH(ip.runtimeVersion), sp.systemName, XR_VERSION_MAJOR(req.minApiVersionSupported),
           XR_VERSION_MINOR(req.minApiVersionSupported), XR_VERSION_MAJOR(req.maxApiVersionSupported),
           XR_VERSION_MINOR(req.maxApiVersionSupported), x.instance_exts.c_str(), x.device_exts.c_str());
    x.ok = true;
    return true;
}

bool listed(const char *const *list, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!std::strcmp(list[i], name)) return true;
    return false;
}

void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to,
             VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = from, b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo d{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    d.imageMemoryBarrierCount = 1, d.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(cb, &d);
}

// v rotated by the unit quaternion q (x, y, z, w)
void rotate(const float q[4], const float v[3], float out[3]) {
    const float x = q[0], y = q[1], z = q[2], w = q[3];
    const float tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    out[0] = v[0] + w * tx + (y * tz - z * ty);
    out[1] = v[1] + w * ty + (z * tx - x * tz);
    out[2] = v[2] + w * tz + (x * ty - y * tx);
}

// The frustum the game draws with for the eyes' lens on a picture of ratio `ratio`: centred (the game's projection is
// symmetric, and the projection layer is told this frustum, so the compositor puts each pixel where it belongs), the
// shorter side grown so nothing the eyes see is left out
Lens frustum(const Lens &eyes, float ratio) {
    const float side = std::fmax(-eyes.l, eyes.r), up = std::fmax(eyes.u, -eyes.d);
    const float v = std::fmax(up, side / ratio), h = v * ratio;
    Lens f;
    f.l = -h, f.r = h, f.u = v, f.d = -v;
    return f;
}

bool is_srgb(VkFormat f) {
    return f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_A8B8G8R8_SRGB_PACK32;
}

// The three boxes at the swapchain's size, in its channel order and the window's encoding: a blit between two sRGB or
// two UNORM formats keeps the bytes, so the copy into the sRGB swapchain shows them as the window does
bool make_boxes(VkFormat window) {
    Xr &x = xr();
    const bool srgb = is_srgb(window);
    const VkFormat format = x.rgba ? (srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM)
                                   : (srgb ? VK_FORMAT_B8G8R8A8_SRGB : VK_FORMAT_B8G8R8A8_UNORM);
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(x.phys, format, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
        rt_log("vr: no blit into format %d: the headset shows nothing", (int)format);
        return false;
    }
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(x.phys, &mem);
    for (int k = 0; k < 9; k++) {   // the pictures' three, the HUD's, the right eye's
        VkImage &box = k < 3 ? x.boxes[k] : k < 6 ? x.hud_boxes[k - 3] : x.boxes_r[k - 6];
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = format;
        ci.extent = {x.width, x.height, 1};
        ci.mipLevels = ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = VK_IMAGE_TILING_OPTIMAL;
        ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        if (vkCreateImage(x.device, &ci, nullptr, &box) != VK_SUCCESS) return false;
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(x.device, box, &mr);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mem.memoryTypeCount && type == UINT32_MAX; i++)
            if ((mr.memoryTypeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                type = i;
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = mr.size, ai.memoryTypeIndex = type;
        VkDeviceMemory dm = VK_NULL_HANDLE;
        if (type == UINT32_MAX || vkAllocateMemory(x.device, &ai, nullptr, &dm) != VK_SUCCESS ||
            vkBindImageMemory(x.device, box, dm, 0) != VK_SUCCESS)
            return false;
    }
    rt_log("vr: VR's pictures: %ux%u, format %d (the window's %d)", x.width, x.height, (int)format, (int)window);
    return true;
}

// A device-local 2D image (one level, one layer) and a view of it; UNDEFINED until the caller's barrier
bool make_image(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkImage &image, VkImageView &view,
                VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT) {
    Xr &x = xr();
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D, ci.format = format, ci.extent = {w, h, 1};
    ci.mipLevels = ci.arrayLayers = 1, ci.samples = VK_SAMPLE_COUNT_1_BIT, ci.tiling = VK_IMAGE_TILING_OPTIMAL, ci.usage = usage;
    if (vkCreateImage(x.device, &ci, nullptr, &image) != VK_SUCCESS) return false;
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(x.device, image, &mr);
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(x.phys, &mem);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < mem.memoryTypeCount && type == UINT32_MAX; i++)
        if ((mr.memoryTypeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) type = i;
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size, ai.memoryTypeIndex = type;
    VkDeviceMemory dm = VK_NULL_HANDLE;
    if (type == UINT32_MAX || vkAllocateMemory(x.device, &ai, nullptr, &dm) != VK_SUCCESS ||
        vkBindImageMemory(x.device, image, dm, 0) != VK_SUCCESS)
        return false;
    VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vc.image = image, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = format, vc.subresourceRange = {aspect, 0, 1, 0, 1};
    return vkCreateImageView(x.device, &vc, nullptr, &view) == VK_SUCCESS;
}

// The HUD panel's shader (shaders/xr_hud.comp): two sampled float images in, two RGBA8 storage images out, push descriptors
bool make_hud_pipeline() {
    Xr &x = xr();
    x.push_set = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(x.device, "vkCmdPushDescriptorSetKHR");
    if (!x.push_set) return false;
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(x.device, &si, nullptr, &x.nearest) != VK_SUCCESS) return false;
    const VkDescriptorSetLayoutBinding b[4] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                               {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                               {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                               {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, li.bindingCount = 4, li.pBindings = b;
    if (vkCreateDescriptorSetLayout(x.device, &li, nullptr, &x.hud_set) != VK_SUCCESS) return false;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 3};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1, pi.pSetLayouts = &x.hud_set, pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(x.device, &pi, nullptr, &x.hud_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = sizeof xr_hud_spv, mi.pCode = xr_hud_spv;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(x.device, &mi, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = x.hud_layout;
    const VkResult r = vkCreateComputePipelines(x.device, VK_NULL_HANDLE, 1, &ci, nullptr, &x.hud_pipe);
    vkDestroyShaderModule(x.device, m, nullptr);
    return r == VK_SUCCESS;
}

bool hud_wanted() { return strcmp(cfg_value("NX2NAVI_VR_HUD"), "panel") == 0; }

// (vk_xr_capture, hud_mu held) the scene without the HUD (with it: keep_hud) and the HUD's panel made from the target as
// it is now and the scene kept from before the HUD; false when there is nothing new to make them from
bool make_panel(VkCommandBuffer cb, bool keep_hud) {
    Xr &x = xr();
    if (!x.scene || !x.hud_target || x.kept == x.used) return false;
    if (!x.hud_tried) {
        x.hud_tried = true;
        if (!make_hud_pipeline()) {
            rt_log("vr: the HUD panel's shader could not be made: the HUD stays in the picture");
            x.hud_pipe = VK_NULL_HANDLE;
        }
    }
    if (!x.hud_pipe) return false;
    if (!x.scene8) {
        const VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (!make_image(x.hud_w, x.hud_h, VK_FORMAT_R8G8B8A8_UNORM, u | VK_IMAGE_USAGE_SAMPLED_BIT, x.scene8, x.scene8_view) ||
            !make_image(x.hud_w, x.hud_h, VK_FORMAT_R8G8B8A8_UNORM, u, x.panel8, x.panel8_view)) {
            x.hud_pipe = VK_NULL_HANDLE;
            return false;
        }
        rt_log("vr: the HUD panel: the HUD's target %ux%u (format %d)", x.hud_w, x.hud_h, (int)x.hud_format);
    }
    if (x.target_viewed != x.hud_target) {   // a view of the game's target, made again when the target is another
        if (x.target_view) vkDestroyImageView(x.device, x.target_view, nullptr);
        VkImageViewCreateInfo vc{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vc.image = x.hud_target, vc.viewType = VK_IMAGE_VIEW_TYPE_2D, vc.format = x.hud_format;
        vc.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        x.target_view = VK_NULL_HANDLE;
        if (vkCreateImageView(x.device, &vc, nullptr, &x.target_view) != VK_SUCCESS) return false;
        x.target_viewed = x.hud_target;
    }
    x.used = x.kept;
    barrier(cb, x.scene8, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    barrier(cb, x.panel8, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, x.hud_pipe);
    const VkDescriptorImageInfo info[4] = {{x.nearest, x.target_view, VK_IMAGE_LAYOUT_GENERAL},
                                           {x.nearest, x.scene_view, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, x.scene8_view, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, x.panel8_view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet w[4]{};
    for (uint32_t i = 0; i < 4; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, w[i].dstBinding = i, w[i].descriptorCount = 1;
        w[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &info[i];
    }
    x.push_set(cb, VK_PIPELINE_BIND_POINT_COMPUTE, x.hud_layout, 0, 4, w);
    const int32_t pc[3] = {(int32_t)x.hud_w, (int32_t)x.hud_h, keep_hud ? 1 : 0};
    vkCmdPushConstants(cb, x.hud_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (x.hud_w + 7) / 8, (x.hud_h + 7) / 8, 1);
    barrier(cb, x.scene8, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    x.panels++;
    return true;
}

// The depth stereo shader (shaders/xr_depth.comp): the scene and its depth sampled, the eye's RGBA8 image out
bool make_depth_pipeline() {
    Xr &x = xr();
    if (!x.push_set) return false;
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(x.device, &si, nullptr, &x.linear) != VK_SUCCESS) return false;
    const VkDescriptorSetLayoutBinding b[3] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                               {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                               {2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, li.bindingCount = 3, li.pBindings = b;
    if (vkCreateDescriptorSetLayout(x.device, &li, nullptr, &x.depth_set) != VK_SUCCESS) return false;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 7};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1, pi.pSetLayouts = &x.depth_set, pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(x.device, &pi, nullptr, &x.depth_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = sizeof xr_depth_spv, mi.pCode = xr_depth_spv;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(x.device, &mi, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = x.depth_layout;
    const VkResult r = vkCreateComputePipelines(x.device, VK_NULL_HANDLE, 1, &ci, nullptr, &x.depth_pipe);
    vkDestroyShaderModule(x.device, m, nullptr);
    return r == VK_SUCCESS;
}

// The depth copy's shader (shaders/xr_depth_keep.comp), made on the thread that records the scene: the scene's depth
// sampled, an R32F storage image out
bool make_keep_pipeline() {
    Xr &x = xr();
    x.keep_push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(x.device, "vkCmdPushDescriptorSetKHR");
    if (!x.keep_push) return false;
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(x.device, &si, nullptr, &x.keep_sampler) != VK_SUCCESS) return false;
    const VkDescriptorSetLayoutBinding b[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
                                               {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, li.bindingCount = 2, li.pBindings = b;
    if (vkCreateDescriptorSetLayout(x.device, &li, nullptr, &x.keep_set) != VK_SUCCESS) return false;
    const VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int32_t) * 2};
    VkPipelineLayoutCreateInfo pi{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pi.setLayoutCount = 1, pi.pSetLayouts = &x.keep_set, pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &pr;
    if (vkCreatePipelineLayout(x.device, &pi, nullptr, &x.keep_layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = sizeof xr_depth_keep_spv, mi.pCode = xr_depth_keep_spv;
    VkShaderModule m = VK_NULL_HANDLE;
    if (vkCreateShaderModule(x.device, &mi, nullptr, &m) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = x.keep_layout;
    const VkResult r = vkCreateComputePipelines(x.device, VK_NULL_HANDLE, 1, &ci, nullptr, &x.keep_pipe);
    vkDestroyShaderModule(x.device, m, nullptr);
    return r == VK_SUCCESS;
}

// (vk_xr_capture, hud_mu held, after make_panel) each eye's picture from the scene make_panel left in scene8 and this
// present's scene depth: the eye half the eyes' distance (NX2NAVI_VR_DEPTH percent of it) to its side of the head's
// view; false without the depth (the head's view then shows the one picture to both eyes)
bool make_eyes(VkCommandBuffer cb, const Drawn &d) {
    Xr &x = xr();
    float planes[2];
    if (!vk_dlaa_vr_planes(planes) || !x.depth_kept) {
        x.stereo_missed++;
        return false;
    }
    if (!x.depth_tried) {
        x.depth_tried = true;
        if (!make_depth_pipeline()) {
            rt_log("vr: the depth stereo shader could not be made: one picture to both eyes");
            x.depth_pipe = VK_NULL_HANDLE;
        }
    }
    if (!x.depth_pipe) return false;
    if (!x.eye8[0] || x.eye8_w != x.hud_w || x.eye8_h != x.hud_h) {   // (another size's are left behind, as the scene's)
        for (int e = 0; e < 2; e++)
            if (!make_image(x.hud_w, x.hud_h, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            x.eye8[e], x.eye8_view[e])) {
                rt_log("vr: the depth stereo eye images %ux%u could not be made: one picture to both eyes", x.hud_w, x.hud_h);
                x.depth_pipe = VK_NULL_HANDLE;
                return false;
            }
        x.eye8_w = x.hud_w, x.eye8_h = x.hud_h;
        rt_log("vr: depth stereo: eye pictures %ux%u", x.hud_w, x.hud_h);
    }
    const float tan_x = d.lens.r * d.scale;   // the game's lens: x_ndc = view x / (z tan_x)
    const float strength = 0.01f * (float)cfg_int("NX2NAVI_VR_DEPTH");
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, x.depth_pipe);
    for (int e = 0; e < 2; e++) {
        barrier(cb, x.eye8[e], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        const VkDescriptorImageInfo info[3] = {{x.linear, x.scene8_view, VK_IMAGE_LAYOUT_GENERAL},
                                               {x.nearest, x.depth_view, VK_IMAGE_LAYOUT_GENERAL},
                                               {VK_NULL_HANDLE, x.eye8_view[e], VK_IMAGE_LAYOUT_GENERAL}};
        VkWriteDescriptorSet w[3]{};
        for (uint32_t i = 0; i < 3; i++) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, w[i].dstBinding = i, w[i].descriptorCount = 1;
            w[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w[i].pImageInfo = &info[i];
        }
        x.push_set(cb, VK_PIPELINE_BIND_POINT_COMPUTE, x.depth_layout, 0, 3, w);
        const float side = (e ? 0.5f : -0.5f) * d.ipd * strength;   // the left eye to the left: the scene moves right in it
        // nothing nearer than 30 cm moves more than it would there: the gaps beside the nearest things stay narrow
        const float pc[7] = {(float)x.hud_w, (float)x.hud_h, side / tan_x, planes[0], planes[1], 0.08f, 1.0f / 0.3f};
        vkCmdPushConstants(cb, x.depth_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vkCmdDispatch(cb, (x.hud_w + 7) / 8, (x.hud_h + 7) / 8, 1);
        barrier(cb, x.eye8[e], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    }
    x.stereo_pictures++;
    return true;
}

// The session's thread: its events, the headset's frames, and each newer box into the swapchain
void run(void *) {
    Xr &x = xr();
    XrGraphicsBindingVulkanKHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
    binding.instance = x.vk_instance;
    binding.physicalDevice = x.phys;
    binding.device = x.device;
    binding.queueFamilyIndex = x.family;
    binding.queueIndex = x.queue_index;
    XrSessionCreateInfo sci{XR_TYPE_SESSION_CREATE_INFO};
    sci.next = &binding;
    sci.systemId = x.system;
    XrSession session = XR_NULL_HANDLE;
    XrResult r = x.xrCreateSession(x.instance, &sci, &session);
    if (XR_FAILED(r)) {
        rt_log("vr: xrCreateSession failed (%d): VR off", (int)r);
        return;
    }
    XrReferenceSpaceCreateInfo rs{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    rs.poseInReferenceSpace.orientation.w = 1.0f;
    XrSpace space = XR_NULL_HANDLE, view_space = XR_NULL_HANDLE;
    x.xrCreateReferenceSpace(session, &rs, &space);
    rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    x.xrCreateReferenceSpace(session, &rs, &view_space);

    uint32_t n = 0;
    x.xrEnumerateSwapchainFormats(session, 0, &n, nullptr);
    std::vector<int64_t> formats(n);
    x.xrEnumerateSwapchainFormats(session, n, &n, formats.data());
    int64_t format = 0;
    for (int64_t want : {(int64_t)VK_FORMAT_R8G8B8A8_SRGB, (int64_t)VK_FORMAT_B8G8R8A8_SRGB})
        for (int64_t f : formats)
            if (!format && f == want) format = f;
    if (!format) {
        rt_log("vr: the runtime offers no 8-bit sRGB swapchain format: VR off");
        return;
    }
    x.rgba.store(format == VK_FORMAT_R8G8B8A8_SRGB);

    XrSwapchain swapchain = XR_NULL_HANDLE, hud_swapchain = XR_NULL_HANDLE;   // (the HUD's panel's own)
    std::vector<XrSwapchainImageVulkanKHR> images, hud_images;
    bool hud_has = false;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    // the swapchain is the window's size, known at the first capture
    const auto make_swapchain = [&]() -> bool {
        uint32_t w = x.want_w.load(), h = x.want_h.load();
        if (!w || !h) return false;
        const uint32_t most = w > h ? w : h;
        if (most > 3840) w = w * 3840 / most, h = h * 3840 / most;
        x.width = w & ~1u, x.height = h & ~1u;
        XrSwapchainCreateInfo sw{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        sw.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
        sw.format = format;
        sw.sampleCount = 1;
        sw.width = 2 * x.width, sw.height = x.height;   // the left eye's picture on the left, the right eye's beside it
        sw.faceCount = sw.arraySize = sw.mipCount = 1;
        r = x.xrCreateSwapchain(session, &sw, &swapchain);
        if (XR_FAILED(r)) {
            rt_log("vr: xrCreateSwapchain failed (%d): VR off", (int)r);
            return false;
        }
        x.xrEnumerateSwapchainImages(swapchain, 0, &n, nullptr);
        images.assign(n, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        x.xrEnumerateSwapchainImages(swapchain, n, &n, (XrSwapchainImageBaseHeader *)images.data());
        sw.width = x.width;
        if (XR_SUCCEEDED(x.xrCreateSwapchain(session, &sw, &hud_swapchain))) {
            uint32_t hn = 0;
            x.xrEnumerateSwapchainImages(hud_swapchain, 0, &hn, nullptr);
            hud_images.assign(hn, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
            x.xrEnumerateSwapchainImages(hud_swapchain, hn, &hn, (XrSwapchainImageBaseHeader *)hud_images.data());
        } else {
            hud_swapchain = XR_NULL_HANDLE;
            rt_log("vr: no swapchain for the HUD's panel: the HUD stays in the picture");
        }
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pi.queueFamilyIndex = x.family;
        vkCreateCommandPool(x.device, &pi, nullptr, &pool);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pool, ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, ai.commandBufferCount = 1;
        vkAllocateCommandBuffers(x.device, &ai, &cb);
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        vkCreateFence(x.device, &fi, nullptr, &fence);
        x.ready.store(true, std::memory_order_release);
        rt_log("vr: swapchain made: %ux%u, %u images, format %lld, queue %u of family %u; the screen %.2f m wide at %.2f m",
               x.width, x.height, n, (long long)format, x.queue_index, x.family, x.screen_width, x.screen_distance);
        return true;
    };
    rt_log("vr: session made");

    bool running = false, have_image = false;
    uint64_t shown = 0, frames = 0, copies = 0, tracked_frames = 0;
    Drawn shown_drawn;   // what the picture in the swapchain was drawn from
    uint64_t last_log = plat_now_ns();
    for (;;) {
        XrEventDataBuffer ev{XR_TYPE_EVENT_DATA_BUFFER};
        bool quit = false;
        while (x.xrPollEvent(x.instance, &ev) == XR_SUCCESS) {
            if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                const auto *s = (const XrEventDataSessionStateChanged *)&ev;
                rt_log("vr: session state %d", (int)s->state);
                if (s->state == XR_SESSION_STATE_READY) {
                    XrSessionBeginInfo bi{XR_TYPE_SESSION_BEGIN_INFO};
                    bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    r = x.xrBeginSession(session, &bi);
                    running = XR_SUCCEEDED(r);
                    if (!running) rt_log("vr: xrBeginSession failed (%d)", (int)r);
                } else if (s->state == XR_SESSION_STATE_STOPPING) {
                    x.xrEndSession(session);
                    running = false;
                } else if (s->state == XR_SESSION_STATE_EXITING || s->state == XR_SESSION_STATE_LOSS_PENDING) {
                    quit = true;
                }
            } else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
                quit = true;
            }
            ev = XrEventDataBuffer{XR_TYPE_EVENT_DATA_BUFFER};
        }
        if (quit) {
            rt_log("vr: the session ended: VR off until the game starts again");
            x.ready.store(false), x.running.store(false);
            std::lock_guard<std::mutex> lk(x.head_mu);
            x.have_head = false;
            return;
        }
        x.running.store(running);
        if (!running) {
            plat_sleep_ms(50);
            continue;
        }
        if (!swapchain && !make_swapchain() && x.want_w.load()) return;   // known and failed: said above
        XrFrameWaitInfo wi{XR_TYPE_FRAME_WAIT_INFO};
        XrFrameState fs{XR_TYPE_FRAME_STATE};
        if (XR_FAILED(x.xrWaitFrame(session, &wi, &fs))) continue;
        XrFrameBeginInfo bfi{XR_TYPE_FRAME_BEGIN_INFO};
        x.xrBeginFrame(session, &bfi);
        frames++;

        {   // the head and the eyes' lenses at this frame's predicted time: what the game draws its next frame with
            XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
            li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            li.displayTime = fs.predictedDisplayTime;
            li.space = space;
            XrViewState vs{XR_TYPE_VIEW_STATE};
            XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}};
            uint32_t count = 0;
            XrSpaceLocation head{XR_TYPE_SPACE_LOCATION};
            const bool views_ok = XR_SUCCEEDED(x.xrLocateViews(session, &li, &vs, 2, &count, views)) && count == 2;
            const bool head_ok = XR_SUCCEEDED(x.xrLocateSpace(view_space, space, fs.predictedDisplayTime, &head)) &&
                                 (head.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);
            if (views_ok && head_ok) {
                Lens eyes;
                eyes.l = std::fmin(std::tan(views[0].fov.angleLeft), std::tan(views[1].fov.angleLeft));
                eyes.r = std::fmax(std::tan(views[0].fov.angleRight), std::tan(views[1].fov.angleRight));
                eyes.u = std::fmax(std::tan(views[0].fov.angleUp), std::tan(views[1].fov.angleUp));
                eyes.d = std::fmin(std::tan(views[0].fov.angleDown), std::tan(views[1].fov.angleDown));
                const float dx = views[1].pose.position.x - views[0].pose.position.x,
                            dy = views[1].pose.position.y - views[0].pose.position.y,
                            dz = views[1].pose.position.z - views[0].pose.position.z;
                const float ipd = std::sqrt(dx * dx + dy * dy + dz * dz);
                std::lock_guard<std::mutex> lk(x.head_mu);
                x.eyes = eyes;
                if (ipd > 0.04f && ipd < 0.09f) x.ipd = ipd;
                std::memcpy(x.head.q, &head.pose.orientation, sizeof x.head.q);
                std::memcpy(x.head.p, &head.pose.position, sizeof x.head.p);
                x.have_head = true;
            }
        }

        int slot = -1;
        uint64_t value = 0;
        Drawn drawn;
        if (swapchain) {
            std::lock_guard<std::mutex> lk(x.mu);
            if (x.latest >= 0 && x.box_value[x.latest] > shown)
                slot = x.latest, value = x.box_value[slot], drawn = x.box_drawn[slot], x.reading = slot;
        }
        // a box into the next image of a swapchain on VR's queue, after the present that wrote it (its timeline value)
        const auto copy_into = [&](XrSwapchain target, std::vector<XrSwapchainImageVulkanKHR> &imgs, VkImage from) -> bool {
            uint32_t index = 0;
            XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            wait.timeout = XR_INFINITE_DURATION;
            if (XR_FAILED(x.xrAcquireSwapchainImage(target, &acq, &index)) || XR_FAILED(x.xrWaitSwapchainImage(target, &wait)))
                return false;
            const VkImage dst = imgs[index].image;
            vkResetCommandPool(x.device, pool, 0);
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cb, &bi);
            barrier(cb, dst, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageCopy c{};
            c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.extent = {x.width, x.height, 1};
            vkCmdCopyImage(cb, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            vkEndCommandBuffer(cb);
            VkSemaphoreSubmitInfo w{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            w.semaphore = x.timeline, w.value = value, w.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
            cbi.commandBuffer = cb;
            VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
            si.waitSemaphoreInfoCount = 1, si.pWaitSemaphoreInfos = &w;
            si.commandBufferInfoCount = 1, si.pCommandBufferInfos = &cbi;
            vkQueueSubmit2(x.queue, 1, &si, fence);
            XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
            x.xrReleaseSwapchainImage(target, &rel);
            vkWaitForFences(x.device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkResetFences(x.device, 1, &fence);
            return true;
        };
        if (slot >= 0 && (drawn.hud || drawn.menu) && hud_swapchain)
            hud_has = copy_into(hud_swapchain, hud_images, x.hud_boxes[slot]);
        if (slot >= 0) {
            uint32_t index = 0;
            XrSwapchainImageAcquireInfo acq{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
            XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
            wait.timeout = XR_INFINITE_DURATION;
            if (XR_SUCCEEDED(x.xrAcquireSwapchainImage(swapchain, &acq, &index)) &&
                XR_SUCCEEDED(x.xrWaitSwapchainImage(swapchain, &wait))) {
                const VkImage dst = images[index].image;
                vkResetCommandPool(x.device, pool, 0);
                VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cb, &bi);
                barrier(cb, dst, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                VkImageCopy c{};
                c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                c.extent = {x.width, x.height, 1};
                vkCmdCopyImage(cb, x.boxes[slot], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
                if (drawn.stereo) {
                    c.dstOffset.x = (int32_t)x.width;
                    vkCmdCopyImage(cb, x.boxes_r[slot], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
                }
                barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
                vkEndCommandBuffer(cb);
                VkSemaphoreSubmitInfo w{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
                w.semaphore = x.timeline, w.value = value, w.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
                VkCommandBufferSubmitInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
                cbi.commandBuffer = cb;
                VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
                si.waitSemaphoreInfoCount = 1, si.pWaitSemaphoreInfos = &w;
                si.commandBufferInfoCount = 1, si.pCommandBufferInfos = &cbi;
                vkQueueSubmit2(x.queue, 1, &si, fence);
                XrSwapchainImageReleaseInfo rel{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                x.xrReleaseSwapchainImage(swapchain, &rel);
                vkWaitForFences(x.device, 1, &fence, VK_TRUE, UINT64_MAX);
                vkResetFences(x.device, 1, &fence);
                have_image = true;
                shown = value;
                shown_drawn = drawn;
                copies++;
            }
            std::lock_guard<std::mutex> lk(x.mu);
            x.reading = -1;
        }

        XrCompositionLayerQuad quad{XR_TYPE_COMPOSITION_LAYER_QUAD};
        quad.space = space;
        quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        quad.subImage.swapchain = swapchain;
        quad.subImage.imageRect = {{0, 0}, {(int32_t)x.width, (int32_t)x.height}};
        quad.pose.orientation.w = 1.0f;
        quad.pose.position.z = -x.screen_distance;
        quad.size = {x.screen_width, x.width ? x.screen_width * (float)x.height / (float)x.width : 0};
        // a picture drawn from the head's pose: a projection layer at that pose, the same picture to both eyes, or with
        // depth stereo each eye's own picture at its own place, half the eyes' distance to its side; the compositor turns
        // it with the head between the game's frames
        XrCompositionLayerProjectionView pv[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                                  {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
        for (int e = 0; e < 2; e++) {
            XrCompositionLayerProjectionView &v = pv[e];
            std::memcpy(&v.pose.orientation, shown_drawn.pose.q, sizeof shown_drawn.pose.q);
            float at[3] = {shown_drawn.pose.p[0], shown_drawn.pose.p[1], shown_drawn.pose.p[2]};
            if (shown_drawn.stereo) {
                const float side[3] = {(e ? 0.5f : -0.5f) * shown_drawn.ipd, 0, 0};
                float off[3];
                rotate(shown_drawn.pose.q, side, off);
                for (int c = 0; c < 3; c++) at[c] += off[c];
            }
            std::memcpy(&v.pose.position, at, sizeof at);
            v.fov.angleLeft = std::atan(shown_drawn.lens.l), v.fov.angleRight = std::atan(shown_drawn.lens.r);
            v.fov.angleUp = std::atan(shown_drawn.lens.u), v.fov.angleDown = std::atan(shown_drawn.lens.d);
            v.subImage.swapchain = swapchain;
            v.subImage.imageRect = {{shown_drawn.stereo && e ? (int32_t)x.width : 0, 0}, {(int32_t)x.width, (int32_t)x.height}};
        }
        XrCompositionLayerProjection proj{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        proj.space = space;
        proj.viewCount = 2;
        proj.views = pv;
        // the HUD's panel: held in front of the head (NX2NAVI_VR_HUD_SIZE centimetres wide, a metre away), over the view
        XrCompositionLayerQuad hud{XR_TYPE_COMPOSITION_LAYER_QUAD};
        hud.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
        hud.space = view_space;
        hud.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
        hud.subImage.swapchain = hud_swapchain;
        hud.subImage.imageRect = {{0, 0}, {(int32_t)x.width, (int32_t)x.height}};
        hud.pose.orientation.w = 1.0f;
        hud.pose.position.z = -1.0f;
        {
            const float wide = 0.01f * (float)cfg_int("NX2NAVI_VR_HUD_SIZE");
            hud.size = {wide, x.width ? wide * (float)x.height / (float)x.width : 0};
        }
        // the paused game's menus go on the HUD's panel, over the world as it was
        const bool show_menu = shown_drawn.tracked && shown_drawn.menu && hud_has && hud_swapchain;
        const bool show_hud = shown_drawn.tracked && shown_drawn.hud && hud_has && hud_swapchain;
        const XrCompositionLayerBaseHeader *layers[] = {shown_drawn.tracked ? (const XrCompositionLayerBaseHeader *)&proj
                                                                            : (const XrCompositionLayerBaseHeader *)&quad,
                                                        (const XrCompositionLayerBaseHeader *)&hud};
        if (shown_drawn.tracked) tracked_frames++;
        XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
        ei.displayTime = fs.predictedDisplayTime;
        ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        ei.layerCount = have_image && fs.shouldRender ? (show_hud || show_menu ? 2 : 1) : 0;
        ei.layers = layers;
        r = x.xrEndFrame(session, &ei);
        if (XR_FAILED(r)) rt_log("vr: xrEndFrame failed (%d)", (int)r);
        const uint64_t now = plat_now_ns();
        if (now - last_log > 10000000000ull) {
            rt_log("vr: %llu headset frames (%llu with the head's view), %llu pictures copied in the last 10 s (%u drawn "
                   "from the head, %u HUD panels made, %u in depth stereo, %u without the scene's depth, %u paused); the "
                   "game's camera %u, lens %u", (unsigned long long)frames, (unsigned long long)tracked_frames,
                   (unsigned long long)copies, x.tracked_pictures.exchange(0), x.panels.exchange(0), x.stereo_pictures.exchange(0),
                   x.stereo_missed.exchange(0), x.menu_pictures.exchange(0), x.camera_given.exchange(0), x.lens_given.exchange(0));
            frames = copies = tracked_frames = 0, last_log = now;
        }
    }
}

} // namespace

extern "C" void vk_xr_instance_exts(const char **ext, uint32_t *n, uint32_t cap) {
    if (!init()) return;
    for (const char *name : xr().instance_list) {
        if (listed(ext, *n, name)) continue;
        if (*n >= cap) {
            rt_log("vr: no room for the instance extension %s: VR off", name);
            xr().ok = false;
            return;
        }
        ext[(*n)++] = name;
    }
}

extern "C" VkPhysicalDevice vk_xr_device(VkInstance instance) {
    Xr &x = xr();
    if (!x.ok) return VK_NULL_HANDLE;
    VkPhysicalDevice p = VK_NULL_HANDLE;
    const XrResult r = x.xrGetVulkanGraphicsDeviceKHR(x.instance, x.system, instance, &p);
    if (XR_FAILED(r) || !p) {
        rt_log("vr: xrGetVulkanGraphicsDeviceKHR failed (%d): VR off", (int)r);
        x.ok = false;
        return VK_NULL_HANDLE;
    }
    return x.phys = p;
}

extern "C" void vk_xr_device_picked(VkPhysicalDevice picked) {
    Xr &x = xr();
    if (!x.ok || picked == x.phys) return;
    VkPhysicalDeviceProperties a{}, b{};
    vkGetPhysicalDeviceProperties(picked, &a);
    if (x.phys) vkGetPhysicalDeviceProperties(x.phys, &b);
    rt_log("vr: the headset's runtime draws with %s, the game with %s (not usable there, or chosen by NX2NAVI_GPU_*): "
           "VR off", x.phys ? b.deviceName : "no GPU", a.deviceName);
    x.ok = false;
}

extern "C" void vk_xr_device_exts(const VkExtensionProperties *have, uint32_t n_have, const char **ext, uint32_t *n,
                                  uint32_t cap) {
    Xr &x = xr();
    if (!x.ok) return;
    for (const char *name : x.device_list) {
        bool there = false;
        for (uint32_t i = 0; i < n_have && !there; i++) there = !std::strcmp(have[i].extensionName, name);
        if (!there) rt_log("vr: the device lacks %s, which the runtime asks for", name);
        else if (!listed(ext, *n, name)) {
            if (*n >= cap) {
                rt_log("vr: no room for the device extension %s: VR off", name);
                x.ok = false;
                return;
            }
            ext[(*n)++] = name;
        }
    }
}

extern "C" uint32_t vk_xr_queue(uint32_t family_queues, uint32_t first_free) {
    Xr &x = xr();
    if (!x.ok) return UINT32_MAX;
    if (first_free >= family_queues || first_free >= 4) {
        rt_log("vr: the graphics family has %u queues, none spare for VR: VR off", family_queues);
        x.ok = false;
        return UINT32_MAX;
    }
    return x.queue_index = first_free;
}

extern "C" void vk_xr_device_ready(VkInstance instance, VkPhysicalDevice phys, VkDevice device, uint32_t family) {
    Xr &x = xr();
    if (!x.ok || x.queue_index == UINT32_MAX) return;
    x.vk_instance = instance, x.phys = phys, x.device = device, x.family = family;
    vkGetDeviceQueue(device, family, x.queue_index, &x.queue);
    VkSemaphoreTypeCreateInfo tc{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    tc.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo sc{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sc.pNext = &tc;
    if (vkCreateSemaphore(device, &sc, nullptr, &x.timeline) != VK_SUCCESS) {
        rt_log("vr: no timeline semaphore: VR off");
        x.ok = false;
        return;
    }
    if (!plat_thread_spawn(run, nullptr, "vr session", 0)) {
        rt_log("vr: the session's thread could not start: VR off");
        x.ok = false;
    }
}

extern "C" int vk_xr_head_wanted(void) { return xr().ok && strcmp(cfg_value("NX2NAVI_VR_VIEW"), "head") == 0; }

extern "C" int vk_xr_depth_wanted(void) { return vk_xr_head_wanted() && cfg_int("NX2NAVI_VR_DEPTH") > 0; }

extern "C" void vk_xr_keep_depth(VkCommandBuffer cb, VkImageView depth, uint32_t w, uint32_t h) {
    Xr &x = xr();
    if (!x.ok || !x.head_view.load(std::memory_order_relaxed) || !depth || !w || !h) return;
    std::lock_guard<std::mutex> lk(x.hud_mu);
    if (!x.keep_tried) {
        x.keep_tried = true;
        if (!make_keep_pipeline()) {
            rt_log("vr: the depth stereo's depth copy could not be made: one picture to both eyes");
            x.keep_pipe = VK_NULL_HANDLE;
        }
    }
    if (!x.keep_pipe) return;
    // the replay's commands go through the recorder thread (vk_rec.h), which this file's direct calls bypass: they are
    // made once it has made every command queued before them
    rec_drain();
    bool fresh = false;
    if (!x.depth || x.depth_w != w || x.depth_h != h) {
        x.depth = VK_NULL_HANDLE, x.depth_kept = 0;   // (another size's is left behind, as the scene's)
        if (!make_image(w, h, VK_FORMAT_R32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, x.depth, x.depth_view)) {
            rt_log("vr: the depth stereo's depth image %ux%u could not be made: one picture to both eyes", w, h);
            x.keep_pipe = VK_NULL_HANDLE;
            return;
        }
        x.depth_w = w, x.depth_h = h, fresh = true;
        rt_log("vr: depth stereo keeps the scene's depth: %ux%u", w, h);
    }
    barrier(cb, x.depth, fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, x.keep_pipe);
    const VkDescriptorImageInfo info[2] = {{x.keep_sampler, depth, VK_IMAGE_LAYOUT_GENERAL},
                                           {VK_NULL_HANDLE, x.depth_view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet wr[2]{};
    for (uint32_t i = 0; i < 2; i++) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, wr[i].dstBinding = i, wr[i].descriptorCount = 1;
        wr[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        wr[i].pImageInfo = &info[i];
    }
    x.keep_push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, x.keep_layout, 0, 2, wr);
    const int32_t pc[2] = {(int32_t)w, (int32_t)h};
    vkCmdPushConstants(cb, x.keep_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    barrier(cb, x.depth, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    x.depth_kept++;
}

extern "C" int vk_xr_head_active(void) {
    Xr &x = xr();
    return x.ok && x.head_view.load(std::memory_order_relaxed) && x.running.load(std::memory_order_relaxed);
}

extern "C" int vk_xr_camera(double pos[3], double at[3], float up[3]) {
    Xr &x = xr();
    if (!x.ok || !x.head_view.load(std::memory_order_relaxed) || !x.running.load(std::memory_order_relaxed)) {
        vk_xr_untracked();
        return 0;
    }
    Pose head;
    Lens eyes;
    float ipd;
    {
        std::lock_guard<std::mutex> lk(x.head_mu);
        if (!x.have_head) return 0;
        head = x.head, eyes = x.eyes;
        ipd = x.ipd;
    }
    // the camera's facing made level (its yaw alone, so the horizon stays level whatever the game's camera does): the
    // head's axes are put on it, x right, y up, z back as OpenXR's
    double f[3] = {at[0] - pos[0], at[1] - pos[1], at[2] - pos[2]};
    const double reach = std::sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    const double flat = std::sqrt(f[0] * f[0] + f[2] * f[2]);
    if (!(reach > 1e-6) || !(flat > 1e-6)) return 0;
    const double bx = -f[0] / flat, bz = -f[2] / flat;
    const double axes[3][3] = {{bz, 0, -bx}, {0, 1, 0}, {bx, 0, bz}};   // right, up, back in the world
    const auto world = [&](const float v[3], double out[3]) {
        for (int c = 0; c < 3; c++) out[c] = axes[0][c] * v[0] + axes[1][c] * v[1] + axes[2][c] * v[2];
    };
    static const float ahead[3] = {0, 0, -1}, upward[3] = {0, 1, 0};
    float hf[3], hu[3];
    rotate(head.q, ahead, hf), rotate(head.q, upward, hu);
    double wf[3], wu[3], wp[3];
    world(hf, wf), world(hu, wu), world(head.p, wp);
    for (int c = 0; c < 3; c++) {
        pos[c] += wp[c];
        at[c] = pos[c] + wf[c] * reach;
        up[c] = (float)wu[c];
    }
    std::lock_guard<std::mutex> lk(x.frame_mu);
    x.frame_on = true;
    x.drawn.tracked = true, x.drawn.pose = head, x.drawn.lens = frustum(eyes, x.picture_ratio.load());
    // the game draws a lens this many times the one the headset is told (NX2NAVI_VR_VIEW_SCALE, 100 % by default: the
    // headset's own): more of the world in the same view, everything smaller above 100 %
    x.render_scale = std::fmax(0.5f, std::fmin(2.0f, (float)cfg_int("NX2NAVI_VR_VIEW_SCALE") / 100.0f));
    x.drawn.scale = x.render_scale, x.drawn.ipd = ipd;
    x.camera_given++;
    x.updates++;
    return 1;
}

extern "C" int vk_xr_lens(float *angle) {
    Xr &x = xr();
    std::lock_guard<std::mutex> lk(x.frame_mu);
    if (!x.frame_on) return 0;
    *angle = 2.0f * std::atan(x.drawn.lens.u * x.render_scale);
    x.lens_given++;
    return 1;
}

extern "C" void vk_xr_untracked(void) {
    Xr &x = xr();
    std::lock_guard<std::mutex> lk(x.frame_mu);
    x.frame_on = false;
    x.drawn.tracked = false;
}

extern "C" VkImage vk_xr_scene_image(VkImage target, uint32_t w, uint32_t h, VkFormat format, int *init) {
    Xr &x = xr();
    *init = 0;
    if (!x.ok || !x.head_view.load(std::memory_order_relaxed) || !(hud_wanted() || vk_xr_depth_wanted()) || !target || !w || !h)
        return VK_NULL_HANDLE;
    // other pictures pass through the upscaler too (a 2560x1440 sRGB one): only the one of the presented picture's size
    // is the scene the HUD is drawn over
    if (w != x.pic_w.load(std::memory_order_relaxed) || h != x.pic_h.load(std::memory_order_relaxed)) return VK_NULL_HANDLE;
    std::lock_guard<std::mutex> lk(x.hud_mu);
    if (x.hud_tried && !x.hud_pipe) return VK_NULL_HANDLE;   // the panel could not be made: the HUD stays in the picture
    if (!x.scene || x.hud_w != w || x.hud_h != h || x.hud_format != format) {
        // (images of another size are left behind: the picture's size changes rarely, at a setting)
        x.scene = x.scene8 = x.panel8 = VK_NULL_HANDLE, x.used = x.kept;
        if (!make_image(w, h, format, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, x.scene, x.scene_view)) {
            static int told_fail;
            if (told_fail++ < 4) rt_log("vr: the HUD panel's scene image %ux%u format %d could not be made", w, h, (int)format);
            x.scene = VK_NULL_HANDLE;
            return VK_NULL_HANDLE;
        }
        rt_log("vr: the HUD panel keeps the scene: %ux%u format %d", w, h, (int)format);
        x.hud_w = w, x.hud_h = h, x.hud_format = format;
        *init = 1;
    }
    x.hud_target = target;
    x.kept++;
    return x.scene;
}

extern "C" int vk_xr_wants_capture(int hdr) {
    Xr &x = xr();
    if (!x.ok || !x.timeline) return 0;
    const bool head = vk_xr_head_wanted();
    static bool hooks_on;   // the camera hooks, switched on the first time the head's view is wanted
    if (head && !hooks_on) hooks_on = true, vr_game_start();
    x.head_view.store(head && !menu_is_open(), std::memory_order_relaxed);
    if (hdr) {
        static bool told;
        if (!told) told = true, rt_log("vr: the headset needs HDR off");
        return 0;
    }
    if (!x.ready.load(std::memory_order_acquire)) return 1;   // the window's size is still to be learned (vk_xr_capture)
    if (x.boxes_failed) return 0;
    std::lock_guard<std::mutex> lk(x.mu);
    for (int i = 0; i < 3; i++)
        if (i != x.reading && i != x.latest) return 1;
    return 0;
}

extern "C" void vk_xr_capture(VkCommandBuffer cb, VkImage window, VkExtent2D extent, VkFormat format, VkImage picture,
                              VkImageLayout picture_layout, const int rect[4], VkSemaphoreSubmitInfo *signal) {
    Xr &x = xr();
    *signal = VkSemaphoreSubmitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    if (!extent.width || !extent.height) return;
    const bool have_picture = picture && rect[2] > 0 && rect[3] > 0;
    if (have_picture) {
        x.picture_ratio.store((float)rect[2] / (float)rect[3], std::memory_order_relaxed);
        x.pic_w.store((uint32_t)rect[2], std::memory_order_relaxed), x.pic_h.store((uint32_t)rect[3], std::memory_order_relaxed);
    }
    // the swapchain at the game's picture's size (the head's view shows it over the eyes' whole view, every pixel
    // counts); the boot screen before it, which has none, is not shown
    if (!x.want_w.load() && have_picture) x.want_h.store((uint32_t)rect[3]), x.want_w.store((uint32_t)rect[2]);
    if (!x.ready.load(std::memory_order_acquire)) return;
    if (!x.boxes_made) {
        x.boxes_made = make_boxes(format);
        x.boxes_failed = !x.boxes_made;
        if (!x.boxes_made) return;
    }
    int slot = -1;
    {
        std::lock_guard<std::mutex> lk(x.mu);
        for (int i = 0; i < 3 && slot < 0; i++) {
            const int c = (x.next_slot + i) % 3;
            if (c != x.reading && c != x.latest) slot = c;
        }
        if (slot < 0) return;
        x.next_slot = (slot + 1) % 3;
    }
    // the head's view: the game's picture (without the window's menu, which the screen shows) with the pose the last
    // gameplay camera update put on it; else the window for the screen
    Drawn d;
    x.stale = x.updates.exchange(0) ? 0 : x.stale + 1;
    const bool head = x.head_view.load(std::memory_order_relaxed) && have_picture;
    if (head && x.stale < 3) {
        std::lock_guard<std::mutex> lk(x.frame_mu);
        d = x.drawn;
    }
    if (!head) x.frozen.tracked = false;
    if (head && x.stale >= 3 && x.frozen.tracked) {   // paused: the frozen world, and the game's picture on the HUD's panel
        d = x.frozen;
        d.hud = false, d.menu = true;
        x.box_drawn[slot] = d;
        x.menu_pictures++;
        VkImageBlit b{};
        b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.dstOffsets[1] = {(int32_t)x.width, (int32_t)x.height, 1};
        {
            std::lock_guard<std::mutex> lk(x.hud_mu);
            b.srcOffsets[1] = {(int32_t)x.hud_w, (int32_t)x.hud_h, 1};
            for (int e = 0; e < (d.stereo ? 2 : 1); e++) {
                const VkImage bx = e ? x.boxes_r[slot] : x.boxes[slot];
                barrier(cb, bx, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                vkCmdBlitImage(cb, d.stereo ? x.eye8[e] : x.scene8, VK_IMAGE_LAYOUT_GENERAL, bx,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
                barrier(cb, bx, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            }
        }
        const VkImage hb = x.hud_boxes[slot];
        barrier(cb, hb, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        b.srcOffsets[0] = {rect[0], rect[1], 0};
        b.srcOffsets[1] = {rect[0] + rect[2], rect[1] + rect[3], 1};
        vkCmdBlitImage(cb, picture, picture_layout, hb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
        barrier(cb, hb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        x.pending = slot;
        signal->semaphore = x.timeline;
        signal->value = ++x.next_value;
        signal->stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        return;
    }
    // the HUD apart (NX2NAVI_VR_HUD=panel) and depth stereo (NX2NAVI_VR_DEPTH) start from the scene kept before the HUD:
    // the head's view without the HUD (with it where it stays in the picture), the HUD on its panel, each eye's picture
    const bool panel = hud_wanted(), depth = d.tracked && vk_xr_depth_wanted();
    bool made = false;
    uint32_t hw = 0, hh = 0;
    if (d.tracked && (panel || depth)) {
        std::lock_guard<std::mutex> lk(x.hud_mu);
        made = make_panel(cb, !panel);
        hw = x.hud_w, hh = x.hud_h;
        if (made && depth) d.stereo = make_eyes(cb, d);
    }
    d.hud = made && panel;
    if (d.tracked && made) x.frozen = d;   // its world stays in scene8 and eye8 until the next one made
    x.box_drawn[slot] = d;
    if (d.tracked) x.tracked_pictures++;
    const VkImage box = x.boxes[slot];
    barrier(cb, box, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageBlit b{};
    b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstOffsets[1] = {(int32_t)x.width, (int32_t)x.height, 1};
    if (made) {
        b.srcOffsets[1] = {(int32_t)hw, (int32_t)hh, 1};
        vkCmdBlitImage(cb, d.stereo ? x.eye8[0] : x.scene8, VK_IMAGE_LAYOUT_GENERAL, box, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &b, VK_FILTER_LINEAR);
        if (d.stereo) {
            const VkImage rb = x.boxes_r[slot];
            barrier(cb, rb, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdBlitImage(cb, x.eye8[1], VK_IMAGE_LAYOUT_GENERAL, rb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
            barrier(cb, rb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        }
        if (d.hud) {
            const VkImage hb = x.hud_boxes[slot];
            barrier(cb, hb, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdBlitImage(cb, x.panel8, VK_IMAGE_LAYOUT_GENERAL, hb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
            barrier(cb, hb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        }
    } else if (d.tracked) {
        b.srcOffsets[0] = {rect[0], rect[1], 0};
        b.srcOffsets[1] = {rect[0] + rect[2], rect[1] + rect[3], 1};
        vkCmdBlitImage(cb, picture, picture_layout, box, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    } else {
        b.srcOffsets[1] = {(int32_t)extent.width, (int32_t)extent.height, 1};
        vkCmdBlitImage(cb, window, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, box, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                       VK_FILTER_LINEAR);
    }
    barrier(cb, box, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    x.pending = slot;
    signal->semaphore = x.timeline;
    signal->value = ++x.next_value;
    signal->stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
}

extern "C" void vk_xr_captured(const VkSemaphoreSubmitInfo *signal) {
    Xr &x = xr();
    if (!signal->semaphore || x.pending < 0) return;
    std::lock_guard<std::mutex> lk(x.mu);
    x.box_value[x.pending] = signal->value;
    x.latest = x.pending;
    x.pending = -1;
}
