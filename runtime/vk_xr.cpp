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
    Pose pose;
    Lens lens;
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
    VkImage boxes[3]{};
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
    std::atomic<bool> running{false};
    // the view now (each present): the head's on the game's camera, or the screen (NX2NAVI_VR_VIEW, the menu open)
    std::atomic<bool> head_view{false};
    std::atomic<float> picture_ratio{16.0f / 9.0f};
    // this frame's (frame_mu): taken at the gameplay camera's update (vk_xr_camera), its lens given at the angle's
    // load (vk_xr_lens), kept for the next picture (vk_xr_capture)
    std::mutex frame_mu;
    bool frame_on = false;
    Drawn drawn;
    std::atomic<uint32_t> camera_given{0}, lens_given{0}, tracked_pictures{0};
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

void barrier(VkCommandBuffer cb, VkImage image, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = from, b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
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
    for (VkImage &box : x.boxes) {
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

    XrSwapchain swapchain = XR_NULL_HANDLE;
    std::vector<XrSwapchainImageVulkanKHR> images;
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
        sw.width = x.width, sw.height = x.height;
        sw.faceCount = sw.arraySize = sw.mipCount = 1;
        r = x.xrCreateSwapchain(session, &sw, &swapchain);
        if (XR_FAILED(r)) {
            rt_log("vr: xrCreateSwapchain failed (%d): VR off", (int)r);
            return false;
        }
        x.xrEnumerateSwapchainImages(swapchain, 0, &n, nullptr);
        images.assign(n, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
        x.xrEnumerateSwapchainImages(swapchain, n, &n, (XrSwapchainImageBaseHeader *)images.data());
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
                std::lock_guard<std::mutex> lk(x.head_mu);
                x.eyes = eyes;
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
        // a picture drawn from the head's pose: a projection layer at that pose, the same picture to both eyes; the
        // compositor turns it with the head between the game's frames
        XrCompositionLayerProjectionView pv[2] = {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW},
                                                  {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}};
        for (XrCompositionLayerProjectionView &v : pv) {
            std::memcpy(&v.pose.orientation, shown_drawn.pose.q, sizeof shown_drawn.pose.q);
            std::memcpy(&v.pose.position, shown_drawn.pose.p, sizeof shown_drawn.pose.p);
            v.fov.angleLeft = std::atan(shown_drawn.lens.l), v.fov.angleRight = std::atan(shown_drawn.lens.r);
            v.fov.angleUp = std::atan(shown_drawn.lens.u), v.fov.angleDown = std::atan(shown_drawn.lens.d);
            v.subImage.swapchain = swapchain;
            v.subImage.imageRect = {{0, 0}, {(int32_t)x.width, (int32_t)x.height}};
        }
        XrCompositionLayerProjection proj{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
        proj.space = space;
        proj.viewCount = 2;
        proj.views = pv;
        const XrCompositionLayerBaseHeader *layers[] = {shown_drawn.tracked ? (const XrCompositionLayerBaseHeader *)&proj
                                                                            : (const XrCompositionLayerBaseHeader *)&quad};
        if (shown_drawn.tracked) tracked_frames++;
        XrFrameEndInfo ei{XR_TYPE_FRAME_END_INFO};
        ei.displayTime = fs.predictedDisplayTime;
        ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
        ei.layerCount = have_image && fs.shouldRender ? 1 : 0;
        ei.layers = layers;
        r = x.xrEndFrame(session, &ei);
        if (XR_FAILED(r)) rt_log("vr: xrEndFrame failed (%d)", (int)r);
        const uint64_t now = plat_now_ns();
        if (now - last_log > 10000000000ull) {
            rt_log("vr: %llu headset frames (%llu with the head's view), %llu pictures copied in the last 10 s (%u drawn "
                   "from the head); the game's camera %u, lens %u", (unsigned long long)frames,
                   (unsigned long long)tracked_frames, (unsigned long long)copies, x.tracked_pictures.exchange(0),
                   x.camera_given.exchange(0), x.lens_given.exchange(0));
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
    {
        std::lock_guard<std::mutex> lk(x.head_mu);
        if (!x.have_head) return 0;
        head = x.head, eyes = x.eyes;
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
    x.camera_given++;
    return 1;
}

extern "C" int vk_xr_lens(float *angle) {
    Xr &x = xr();
    std::lock_guard<std::mutex> lk(x.frame_mu);
    if (!x.frame_on) return 0;
    *angle = 2.0f * std::atan(x.drawn.lens.u);
    x.lens_given++;
    return 1;
}

extern "C" void vk_xr_untracked(void) {
    Xr &x = xr();
    std::lock_guard<std::mutex> lk(x.frame_mu);
    x.frame_on = false;
    x.drawn.tracked = false;
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
    if (have_picture) x.picture_ratio.store((float)rect[2] / (float)rect[3], std::memory_order_relaxed);
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
    if (x.head_view.load(std::memory_order_relaxed) && have_picture) {
        std::lock_guard<std::mutex> lk(x.frame_mu);
        d = x.drawn;
    }
    x.box_drawn[slot] = d;
    if (d.tracked) x.tracked_pictures++;
    const VkImage box = x.boxes[slot];
    barrier(cb, box, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkImageBlit b{};
    b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    b.dstOffsets[1] = {(int32_t)x.width, (int32_t)x.height, 1};
    if (d.tracked) {
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
