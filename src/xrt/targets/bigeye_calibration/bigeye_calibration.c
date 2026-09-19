// Copyright 2026, Samuel Rounce
// SPDX-License-Identifier: BSL-1.0
//
// In-headset eye gaze calibration for the Monado bigeye driver. Run with the
// argument "follow" for a demo where a square the size of the foveated
// full-resolution region tracks the reported gaze.
//
// Shows a white fixation quad (view-locked) at known gaze angles, records the
// gaze reported through XR_EXT_eye_gaze_interaction, fits
// reported = gain * true + bias per axis and writes the result to
// ~/.config/monado/bigeye_calibration.json for the driver to invert.
//
// Rendering uses XR_KHR_vulkan_enable2. Each fixation target is a small quad
// composition layer whose swapchain image is simply cleared to white by a
// render pass, so there is no pipeline, vertex data or shader.

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#include <vulkan/vulkan.h>

#define XR_USE_GRAPHICS_API_VULKAN
#define XR_USE_TIMESPEC
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#define CK(expr)                                                                                                       \
	do {                                                                                                           \
		XrResult res_ = (expr);                                                                                \
		if (XR_FAILED(res_)) {                                                                                 \
			fprintf(stderr, "%s failed: %d\n", #expr, res_);                                               \
			exit(1);                                                                                       \
		}                                                                                                      \
	} while (0)

#define VK(expr)                                                                                                       \
	do {                                                                                                           \
		VkResult vr_ = (expr);                                                                                 \
		if (vr_ != VK_SUCCESS) {                                                                               \
			fprintf(stderr, "%s failed: %d\n", #expr, vr_);                                                \
			exit(1);                                                                                       \
		}                                                                                                      \
	} while (0)

#define DEG(x) ((x) * 180.0 / M_PI)
#define RAD(x) ((x) * M_PI / 180.0)

struct target
{
	float yaw_deg;   // positive = right
	float pitch_deg; // positive = up
};

// Cross + corners, ending back at center.
static const struct target targets[] = {
    {0, 0},  {15, 0},  {-15, 0}, {25, 0},   {-25, 0}, {0, 8},    {0, -8},
    {0, 14}, {0, -14}, {15, 8},  {-15, 8},  {15, -8}, {-15, -8}, {0, 0},
};
#define NUM_TARGETS (sizeof(targets) / sizeof(targets[0]))
#define MOVE_NS 700000000L  // 0.7 s glide to the next target, not sampled
#define HOLD_NS 1600000000L // 1.6 s hold at the target, sampled
#define PHASE_NS (MOVE_NS + HOLD_NS)
#define START_DELAY_NS 3000000000L // time to acquire the first target before sampling

struct sample_stats
{
	double sum_yaw, sum_pitch;
	double sq_yaw, sq_pitch;
	int count;
};

static struct sample_stats stats[NUM_TARGETS];

static void
gaze_angles_from_quat(XrQuaternionf q, double *out_yaw_deg, double *out_pitch_deg)
{
	// forward = q * (0, 0, -1)
	double fx = -2.0 * (q.x * q.z + q.w * q.y);
	double fy = 2.0 * (q.w * q.x - q.y * q.z);
	double fz = 2.0 * (q.x * q.x + q.y * q.y) - 1.0;

	*out_yaw_deg = DEG(atan2(fx, -fz));
	*out_pitch_deg = DEG(asin(fy));
}

static XrPosef
quad_pose_from_angles(double yaw_deg, double pitch_deg)
{
	double yaw = RAD(yaw_deg);
	double pitch = RAD(pitch_deg);

	// 1 m away in the target direction, +x right, +y up, -z forward.
	XrPosef pose;
	pose.position.x = (float)(sin(yaw) * cos(pitch));
	pose.position.y = (float)sin(pitch);
	pose.position.z = (float)(-cos(yaw) * cos(pitch));

	// Face the viewer: yaw about +y (looking right needs -yaw), then pitch about +x.
	double hy = -yaw / 2.0, hp = pitch / 2.0;
	XrQuaternionf qy = {0, (float)sin(hy), 0, (float)cos(hy)};
	XrQuaternionf qp = {(float)sin(hp), 0, 0, (float)cos(hp)};
	pose.orientation.w = qy.w * qp.w - qy.x * qp.x - qy.y * qp.y - qy.z * qp.z;
	pose.orientation.x = qy.w * qp.x + qy.x * qp.w + qy.y * qp.z - qy.z * qp.y;
	pose.orientation.y = qy.w * qp.y - qy.x * qp.z + qy.y * qp.w + qy.z * qp.x;
	pose.orientation.z = qy.w * qp.z + qy.x * qp.y - qy.y * qp.x + qy.z * qp.w;

	return pose;
}

// Encode a perceptual sRGB grey level to the linear value a clear on an sRGB
// format attachment expects (the hardware re-applies the sRGB transfer curve).
static float
srgb_to_linear(float s)
{
	return s <= 0.04045f ? s / 12.92f : powf((s + 0.055f) / 1.055f, 2.4f);
}

static float
grey_env(const char *name, float def_srgb)
{
	const char *v = getenv(name);
	float s = v != NULL ? (float)atof(v) : def_srgb;
	return srgb_to_linear(s);
}

// Maps raw driver output (yaw, pitch) to true gaze angles with
// true = c0 + c1*y + c2*p + c3*y*p. The cross term handles the model's axis
// coupling; squared terms were tried and extrapolate badly once the raw
// output drifts between sessions.
#define POLY_TERMS 4

static void
poly_terms(double y, double p, double t[POLY_TERMS])
{
	t[0] = 1;
	t[1] = y;
	t[2] = p;
	t[3] = y * p;
}

static double
poly_eval(const double c[POLY_TERMS], double y, double p)
{
	double t[POLY_TERMS];
	poly_terms(y, p, t);
	double r = 0;
	for (int i = 0; i < POLY_TERMS; i++) {
		r += c[i] * t[i];
	}
	return r;
}

// Least squares via normal equations, Gaussian elimination with pivoting.
static bool
fit_poly(const double *ry, const double *rp, const double *truth, int n, double c[POLY_TERMS])
{
	double a[POLY_TERMS][POLY_TERMS + 1] = {{0}};
	for (int k = 0; k < n; k++) {
		double t[POLY_TERMS];
		poly_terms(ry[k], rp[k], t);
		for (int i = 0; i < POLY_TERMS; i++) {
			for (int j = 0; j < POLY_TERMS; j++) {
				a[i][j] += t[i] * t[j];
			}
			a[i][POLY_TERMS] += t[i] * truth[k];
		}
	}
	for (int i = 0; i < POLY_TERMS; i++) {
		int piv = i;
		for (int r = i + 1; r < POLY_TERMS; r++) {
			if (fabs(a[r][i]) > fabs(a[piv][i])) {
				piv = r;
			}
		}
		if (fabs(a[piv][i]) < 1e-12) {
			return false;
		}
		for (int j = 0; j <= POLY_TERMS; j++) {
			double tmp = a[i][j];
			a[i][j] = a[piv][j];
			a[piv][j] = tmp;
		}
		for (int r = 0; r < POLY_TERMS; r++) {
			if (r == i) {
				continue;
			}
			double f = a[r][i] / a[i][i];
			for (int j = i; j <= POLY_TERMS; j++) {
				a[r][j] -= f * a[i][j];
			}
		}
	}
	for (int i = 0; i < POLY_TERMS; i++) {
		c[i] = a[i][POLY_TERMS] / a[i][i];
	}
	return true;
}

/*
 *
 * Vulkan, created through XR_KHR_vulkan_enable2.
 *
 */

struct vk_state
{
	VkInstance instance;
	VkPhysicalDevice phys;
	VkDevice device;
	uint32_t queue_family;
	VkQueue queue;
	VkCommandPool cmd_pool;
	VkRenderPass render_pass;
	VkFormat format;
};

static void
vk_init(XrInstance xr_instance, XrSystemId system_id, struct vk_state *vk)
{
	PFN_xrCreateVulkanInstanceKHR create_instance = NULL;
	PFN_xrGetVulkanGraphicsDevice2KHR get_phys = NULL;
	PFN_xrCreateVulkanDeviceKHR create_device = NULL;
	PFN_xrGetVulkanGraphicsRequirements2KHR get_reqs = NULL;
	xrGetInstanceProcAddr(xr_instance, "xrCreateVulkanInstanceKHR", (PFN_xrVoidFunction *)&create_instance);
	xrGetInstanceProcAddr(xr_instance, "xrGetVulkanGraphicsDevice2KHR", (PFN_xrVoidFunction *)&get_phys);
	xrGetInstanceProcAddr(xr_instance, "xrCreateVulkanDeviceKHR", (PFN_xrVoidFunction *)&create_device);
	xrGetInstanceProcAddr(xr_instance, "xrGetVulkanGraphicsRequirements2KHR", (PFN_xrVoidFunction *)&get_reqs);

	XrGraphicsRequirementsVulkan2KHR reqs = {.type = XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
	CK(get_reqs(xr_instance, system_id, &reqs));

	VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
	                         .pApplicationName = "bigeye_calibration",
	                         .apiVersion = VK_API_VERSION_1_1};
	VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
	XrVulkanInstanceCreateInfoKHR xr_ici = {.type = XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR,
	                                        .systemId = system_id,
	                                        .pfnGetInstanceProcAddr = vkGetInstanceProcAddr,
	                                        .vulkanCreateInfo = &ici};
	VkResult vk_res;
	CK(create_instance(xr_instance, &xr_ici, &vk->instance, &vk_res));
	VK(vk_res);

	XrVulkanGraphicsDeviceGetInfoKHR gdi = {.type = XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR,
	                                        .systemId = system_id,
	                                        .vulkanInstance = vk->instance};
	CK(get_phys(xr_instance, &gdi, &vk->phys));

	// Pick a graphics queue family.
	uint32_t qf_count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(vk->phys, &qf_count, NULL);
	VkQueueFamilyProperties *qf = calloc(qf_count, sizeof(*qf));
	vkGetPhysicalDeviceQueueFamilyProperties(vk->phys, &qf_count, qf);
	vk->queue_family = 0;
	for (uint32_t i = 0; i < qf_count; i++) {
		if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			vk->queue_family = i;
			break;
		}
	}
	free(qf);

	float prio = 1.0f;
	VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
	                               .queueFamilyIndex = vk->queue_family,
	                               .queueCount = 1,
	                               .pQueuePriorities = &prio};
	VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
	                          .queueCreateInfoCount = 1,
	                          .pQueueCreateInfos = &qci};
	XrVulkanDeviceCreateInfoKHR xr_dci = {.type = XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR,
	                                      .systemId = system_id,
	                                      .pfnGetInstanceProcAddr = vkGetInstanceProcAddr,
	                                      .vulkanPhysicalDevice = vk->phys,
	                                      .vulkanCreateInfo = &dci};
	CK(create_device(xr_instance, &xr_dci, &vk->device, &vk_res));
	VK(vk_res);

	vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

	VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
	                               .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
	                               .queueFamilyIndex = vk->queue_family};
	VK(vkCreateCommandPool(vk->device, &pci, NULL, &vk->cmd_pool));

	// Render pass that just clears the color attachment.
	vk->format = VK_FORMAT_R8G8B8A8_SRGB;
	VkAttachmentDescription att = {.format = vk->format,
	                               .samples = VK_SAMPLE_COUNT_1_BIT,
	                               .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
	                               .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
	                               .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
	                               .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
	                               .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	                               .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkAttachmentReference ref = {.attachment = 0, .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkSubpassDescription sub = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
	                            .colorAttachmentCount = 1,
	                            .pColorAttachments = &ref};
	VkRenderPassCreateInfo rpci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
	                               .attachmentCount = 1,
	                               .pAttachments = &att,
	                               .subpassCount = 1,
	                               .pSubpasses = &sub};
	VK(vkCreateRenderPass(vk->device, &rpci, NULL, &vk->render_pass));
}


// A one-colour swapchain: each image is pre-recorded to clear to `color`.
struct solid_swapchain
{
	XrSwapchain swapchain;
	uint32_t img_count;
	VkCommandBuffer *cmds;
};

static void
solid_swapchain_init(XrSession session, struct vk_state *vk, uint32_t size, const float color[4],
                     struct solid_swapchain *out)
{
	XrSwapchainCreateInfo scci = {.type = XR_TYPE_SWAPCHAIN_CREATE_INFO,
	                              .usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT,
	                              .format = vk->format,
	                              .sampleCount = 1,
	                              .width = size,
	                              .height = size,
	                              .faceCount = 1,
	                              .arraySize = 1,
	                              .mipCount = 1};
	CK(xrCreateSwapchain(session, &scci, &out->swapchain));

	CK(xrEnumerateSwapchainImages(out->swapchain, 0, &out->img_count, NULL));
	XrSwapchainImageVulkan2KHR *images = calloc(out->img_count, sizeof(*images));
	for (uint32_t i = 0; i < out->img_count; i++) {
		images[i].type = XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR;
	}
	CK(xrEnumerateSwapchainImages(out->swapchain, out->img_count, &out->img_count,
	                              (XrSwapchainImageBaseHeader *)images));

	out->cmds = calloc(out->img_count, sizeof(*out->cmds));
	for (uint32_t i = 0; i < out->img_count; i++) {
		VkImageViewCreateInfo vci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		                             .image = images[i].image,
		                             .viewType = VK_IMAGE_VIEW_TYPE_2D,
		                             .format = vk->format,
		                             .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
		                                                  .levelCount = 1,
		                                                  .layerCount = 1}};
		VkImageView view;
		VK(vkCreateImageView(vk->device, &vci, NULL, &view));

		VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
		                               .renderPass = vk->render_pass,
		                               .attachmentCount = 1,
		                               .pAttachments = &view,
		                               .width = size,
		                               .height = size,
		                               .layers = 1};
		VkFramebuffer fb;
		VK(vkCreateFramebuffer(vk->device, &fci, NULL, &fb));

		VkCommandBufferAllocateInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
		                                   .commandPool = vk->cmd_pool,
		                                   .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
		                                   .commandBufferCount = 1};
		VK(vkAllocateCommandBuffers(vk->device, &cbi, &out->cmds[i]));

		VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		VK(vkBeginCommandBuffer(out->cmds[i], &bi));
		VkClearValue clear = {.color = {.float32 = {color[0], color[1], color[2], color[3]}}};
		VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		                             .renderPass = vk->render_pass,
		                             .framebuffer = fb,
		                             .renderArea = {.extent = {size, size}},
		                             .clearValueCount = 1,
		                             .pClearValues = &clear};
		vkCmdBeginRenderPass(out->cmds[i], &rbi, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdEndRenderPass(out->cmds[i]);
		VK(vkEndCommandBuffer(out->cmds[i]));
	}
	free(images);
}

// Acquire, submit the pre-recorded clear, release. Content is constant.
static void
solid_swapchain_present(struct vk_state *vk, struct solid_swapchain *sc)
{
	uint32_t idx;
	CK(xrAcquireSwapchainImage(sc->swapchain, NULL, &idx));
	XrSwapchainImageWaitInfo wi = {.type = XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, .timeout = XR_INFINITE_DURATION};
	CK(xrWaitSwapchainImage(sc->swapchain, &wi));

	VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
	                       .commandBufferCount = 1,
	                       .pCommandBuffers = &sc->cmds[idx]};
	VK(vkQueueSubmit(vk->queue, 1, &submit, VK_NULL_HANDLE));
	VK(vkQueueWaitIdle(vk->queue));

	CK(xrReleaseSwapchainImage(sc->swapchain, NULL));
}

int
main(int argc, char **argv)
{
	bool follow = argc > 1 && strcmp(argv[1], "follow") == 0;
	bool record = argc > 2 && strcmp(argv[1], "record") == 0;
	bool recenter = argc > 1 && strcmp(argv[1], "recenter") == 0;
	const char *record_path = record ? argv[2] : NULL;
	setvbuf(stdout, NULL, _IONBF, 0);

	/*
	 * Instance, system.
	 */
	const char *exts[] = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME, "XR_EXT_eye_gaze_interaction",
	                      XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME};
	XrInstanceCreateInfo ici = {.type = XR_TYPE_INSTANCE_CREATE_INFO,
	                            .applicationInfo = {.applicationName = "bigeye_calib",
	                                                .apiVersion = XR_API_VERSION_1_0},
	                            .enabledExtensionCount = 3,
	                            .enabledExtensionNames = exts};
	XrInstance instance;
	CK(xrCreateInstance(&ici, &instance));

	// Record mode logs CLOCK_MONOTONIC so labels join directly with the
	// driver capture, which stamps frames with the same clock.
	PFN_xrConvertTimeToTimespecTimeKHR to_timespec = NULL;
	xrGetInstanceProcAddr(instance, "xrConvertTimeToTimespecTimeKHR", (PFN_xrVoidFunction *)&to_timespec);

	XrSystemGetInfo sgi = {.type = XR_TYPE_SYSTEM_GET_INFO, .formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY};
	XrSystemId system_id;
	CK(xrGetSystem(instance, &sgi, &system_id));

	XrSystemEyeGazeInteractionPropertiesEXT gaze_props = {
	    .type = XR_TYPE_SYSTEM_EYE_GAZE_INTERACTION_PROPERTIES_EXT};
	XrSystemProperties props = {.type = XR_TYPE_SYSTEM_PROPERTIES, .next = &gaze_props};
	CK(xrGetSystemProperties(instance, system_id, &props));
	if (!gaze_props.supportsEyeGazeInteraction) {
		fprintf(stderr, "System does not support eye gaze interaction\n");
		return 1;
	}

	struct vk_state vk = {0};
	vk_init(instance, system_id, &vk);

	XrGraphicsBindingVulkan2KHR vk_binding = {.type = XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR,
	                                          .instance = vk.instance,
	                                          .physicalDevice = vk.phys,
	                                          .device = vk.device,
	                                          .queueFamilyIndex = vk.queue_family,
	                                          .queueIndex = 0};
	XrSessionCreateInfo sci = {.type = XR_TYPE_SESSION_CREATE_INFO, .next = &vk_binding, .systemId = system_id};
	XrSession session;
	CK(xrCreateSession(instance, &sci, &session));

	/*
	 * Gaze action.
	 */
	XrActionSet action_set;
	XrActionSetCreateInfo asci = {.type = XR_TYPE_ACTION_SET_CREATE_INFO};
	strcpy(asci.actionSetName, "calib");
	strcpy(asci.localizedActionSetName, "calib");
	CK(xrCreateActionSet(instance, &asci, &action_set));

	XrAction gaze_action;
	XrActionCreateInfo aci = {.type = XR_TYPE_ACTION_CREATE_INFO, .actionType = XR_ACTION_TYPE_POSE_INPUT};
	strcpy(aci.actionName, "gaze");
	strcpy(aci.localizedActionName, "gaze");
	CK(xrCreateAction(action_set, &aci, &gaze_action));

	XrPath profile, gaze_path;
	CK(xrStringToPath(instance, "/interaction_profiles/ext/eye_gaze_interaction", &profile));
	CK(xrStringToPath(instance, "/user/eyes_ext/input/gaze_ext/pose", &gaze_path));
	XrActionSuggestedBinding binding = {.action = gaze_action, .binding = gaze_path};
	XrInteractionProfileSuggestedBinding ipsb = {.type = XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING,
	                                             .interactionProfile = profile,
	                                             .countSuggestedBindings = 1,
	                                             .suggestedBindings = &binding};
	CK(xrSuggestInteractionProfileBindings(instance, &ipsb));

	XrSessionActionSetsAttachInfo sasai = {.type = XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO,
	                                       .countActionSets = 1,
	                                       .actionSets = &action_set};
	CK(xrAttachSessionActionSets(session, &sasai));

	XrActionSpaceCreateInfo aspci = {.type = XR_TYPE_ACTION_SPACE_CREATE_INFO,
	                                 .action = gaze_action,
	                                 .poseInActionSpace = {.orientation = {.w = 1}}};
	XrSpace gaze_space;
	CK(xrCreateActionSpace(session, &aspci, &gaze_space));

	XrReferenceSpaceCreateInfo rsci = {.type = XR_TYPE_REFERENCE_SPACE_CREATE_INFO,
	                                   .referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW,
	                                   .poseInReferenceSpace = {.orientation = {.w = 1}}};
	XrSpace view_space;
	CK(xrCreateReferenceSpace(session, &rsci, &view_space));

	/*
	 * Two solid swapchains: a light-grey background filling the view and a
	 * dark-grey fixation square. Low contrast is easier on the eyes.
	 */
	// Perceptual sRGB greys (override with BIGEYE_CALIB_BG / _FG, 0..1).
	float bg = grey_env("BIGEYE_CALIB_BG", 0.35f);
	float fg = grey_env("BIGEYE_CALIB_FG", 0.15f);
	const float bg_color[4] = {bg, bg, bg, 1.0f};
	const float fg_color[4] = {fg, fg, fg, 1.0f};
	// A small red dot in the middle of the square gives a precise fixation point.
	const float dot_color[4] = {srgb_to_linear(0.85f), srgb_to_linear(0.12f), srgb_to_linear(0.12f), 1.0f};
	struct solid_swapchain bg_sc, fg_sc, dot_sc;
	solid_swapchain_init(session, &vk, 64, bg_color, &bg_sc);
	solid_swapchain_init(session, &vk, 128, fg_color, &fg_sc);
	solid_swapchain_init(session, &vk, 16, dot_color, &dot_sc);
	const float ref_color[4] = {srgb_to_linear(0.9f), srgb_to_linear(0.9f), srgb_to_linear(0.9f), 1.0f};
	struct solid_swapchain ref_sc;
	solid_swapchain_init(session, &vk, 16, ref_color, &ref_sc);

	/*
	 * Wait for the session to become ready.
	 */
	bool running = false;
	while (!running) {
		XrEventDataBuffer ev = {.type = XR_TYPE_EVENT_DATA_BUFFER};
		while (xrPollEvent(instance, &ev) == XR_SUCCESS) {
			if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
				XrEventDataSessionStateChanged *ssc = (XrEventDataSessionStateChanged *)&ev;
				if (ssc->state == XR_SESSION_STATE_READY) {
					XrSessionBeginInfo sbi = {
					    .type = XR_TYPE_SESSION_BEGIN_INFO,
					    .primaryViewConfigurationType =
					        XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO};
					CK(xrBeginSession(session, &sbi));
					running = true;
				}
			}
			ev.type = XR_TYPE_EVENT_DATA_BUFFER;
		}
	}

	if (!follow && !record && !recenter) {
		// Measure the raw mapping: move any current calibration aside and let
		// the driver's hot reload drop it before the first target.
		const char *home_dir = getenv("HOME");
		char cur[600], bak[608];
		snprintf(cur, sizeof(cur), "%s/.config/monado/bigeye_calibration.json", home_dir);
		snprintf(bak, sizeof(bak), "%s.bak", cur);
		if (rename(cur, bak) == 0) {
			printf("Previous calibration kept at %s\n", bak);
		}
		printf("Follow the dark square with your eyes, keep your head still.\n");
	}

	/*
	 * Recenter mode: five seconds on the centre dot, then the mean reported
	 * gaze is stored as an offset in the calibration file. Corrects the
	 * session-to-session drift from headset fit without a full calibration.
	 */
	if (recenter) {
		XrActiveActionSet active_c = {.actionSet = action_set};
		XrActionsSyncInfo asi_c = {.type = XR_TYPE_ACTIONS_SYNC_INFO,
		                           .countActiveActionSets = 1,
		                           .activeActionSets = &active_c};
		XrTime start = 0;
		double sum_y = 0, sum_p = 0;
		int count = 0;
		printf("Recenter: look at the dot.\n");
		while (true) {
			XrEventDataBuffer ev = {.type = XR_TYPE_EVENT_DATA_BUFFER};
			while (xrPollEvent(instance, &ev) == XR_SUCCESS) {
				ev.type = XR_TYPE_EVENT_DATA_BUFFER;
			}
			XrFrameState fs = {.type = XR_TYPE_FRAME_STATE};
			CK(xrWaitFrame(session, NULL, &fs));
			CK(xrBeginFrame(session, NULL));
			if (start == 0) {
				start = fs.predictedDisplayTime;
			}
			double t = (fs.predictedDisplayTime - start) / 1e9;
			if (t > 5.0) {
				XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
				                      .displayTime = fs.predictedDisplayTime,
				                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE};
				CK(xrEndFrame(session, &fei));
				break;
			}
			CK(xrSyncActions(session, &asi_c));
			XrSpaceLocation loc = {.type = XR_TYPE_SPACE_LOCATION};
			if (t > 2.0 &&
			    XR_SUCCEEDED(xrLocateSpace(gaze_space, view_space, fs.predictedDisplayTime, &loc)) &&
			    (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)) {
				double gy, gp;
				gaze_angles_from_quat(loc.pose.orientation, &gy, &gp);
				sum_y += gy;
				sum_p += gp;
				count++;
			}
			solid_swapchain_present(&vk, &bg_sc);
			solid_swapchain_present(&vk, &dot_sc);
			XrCompositionLayerQuad bg_quad = {
			    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
			    .space = view_space,
			    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
			    .subImage = {.swapchain = bg_sc.swapchain, .imageRect = {.extent = {64, 64}}},
			    .pose = {.orientation = {.w = 1}, .position = {0, 0, -2.0f}},
			    .size = {8.0f, 8.0f},
			};
			XrCompositionLayerQuad dot = {
			    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
			    .space = view_space,
			    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
			    .subImage = {.swapchain = dot_sc.swapchain, .imageRect = {.extent = {16, 16}}},
			    .pose = quad_pose_from_angles(0, 0),
			    .size = {0.015f, 0.015f},
			};
			const XrCompositionLayerBaseHeader *layers[] = {(XrCompositionLayerBaseHeader *)&bg_quad,
			                                                (XrCompositionLayerBaseHeader *)&dot};
			XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
			                      .displayTime = fs.predictedDisplayTime,
			                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
			                      .layerCount = fs.shouldRender ? 2 : 0,
			                      .layers = layers};
			CK(xrEndFrame(session, &fei));
		}
		if (count < 30) {
			fprintf(stderr, "gaze not tracked\n");
			return 1;
		}
		double off_y = sum_y / count, off_p = sum_p / count;
		printf("centre read yaw %+.2f pitch %+.2f, storing as offset\n", off_y, off_p);

		// Rewrite the calibration file with the offsets, keeping the mapping.
		const char *home = getenv("HOME");
		char path[600];
		snprintf(path, sizeof(path), "%s/.config/monado/bigeye_calibration.json", home);
		FILE *f = fopen(path, "r");
		char buf[4096] = {0};
		if (f != NULL) {
			size_t n = fread(buf, 1, sizeof(buf) - 1, f);
			buf[n] = 0;
			fclose(f);
		}
		// The residual was measured with the stored offsets applied, so add.
		double prev_y = 0, prev_p = 0;
		char *py = strstr(buf, "\"yaw_offset\":");
		char *pp = strstr(buf, "\"pitch_offset\":");
		if (py != NULL) {
			prev_y = atof(py + strlen("\"yaw_offset\":"));
		}
		if (pp != NULL) {
			prev_p = atof(pp + strlen("\"pitch_offset\":"));
		}
		off_y += prev_y;
		off_p += prev_p;
		printf("total offset yaw %+.2f pitch %+.2f\n", off_y, off_p);

		// Strip any previous offset lines and the closing brace.
		char *cut = strstr(buf, "\t\"yaw_offset\"");
		if (cut == NULL) {
			cut = strrchr(buf, '}');
		}
		if (cut != NULL) {
			*cut = 0;
		}
		size_t len = strlen(buf);
		while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == ',' || buf[len - 1] == ' ')) {
			buf[--len] = 0;
		}
		f = fopen(path, "w");
		if (f == NULL) {
			fprintf(stderr, "cannot write %s\n", path);
			return 1;
		}
		if (len > 1) {
			fprintf(f, "%s,\n", buf);
		} else {
			fprintf(f, "{\n");
		}
		fprintf(f, "\t\"yaw_offset\": %.3f,\n\t\"pitch_offset\": %.3f\n}\n", off_y, off_p);
		fclose(f);
		printf("wrote %s\n", path);
		xrDestroySession(session);
		xrDestroyInstance(instance);
		return 0;
	}


	/*
	 * Record mode: the dot does slow smooth pursuit over the gaze range while
	 * the target angles are logged per frame with the display timestamp, so
	 * they can be joined against the driver's BIGEYE_CAPTURE frames to train
	 * a per-user model. Ends with an eyes-closed segment for lid labels.
	 */
	if (record) {
		const char *secs_env = getenv("BIGEYE_RECORD_SECONDS");
		double pursuit_s = secs_env != NULL ? atof(secs_env) : 120.0;
		const double lead_s = 3.0, blink_s = 15.0;
		FILE *log = fopen(record_path, "w");
		if (log == NULL) {
			fprintf(stderr, "cannot write %s\n", record_path);
			return 1;
		}
		XrActiveActionSet active_r = {.actionSet = action_set};
		XrActionsSyncInfo asi_r = {.type = XR_TYPE_ACTIONS_SYNC_INFO,
		                           .countActiveActionSets = 1,
		                           .activeActionSets = &active_r};
		XrTime start = 0;
		printf("Record mode: %.0f s pursuit then %.0f s eyes closed.\n", pursuit_s, blink_s);
		printf("Follow the red dot smoothly. When it disappears, close your eyes until the run ends.\n");

		while (true) {
			XrEventDataBuffer ev = {.type = XR_TYPE_EVENT_DATA_BUFFER};
			while (xrPollEvent(instance, &ev) == XR_SUCCESS) {
				ev.type = XR_TYPE_EVENT_DATA_BUFFER;
			}
			XrFrameState fs = {.type = XR_TYPE_FRAME_STATE};
			CK(xrWaitFrame(session, NULL, &fs));
			CK(xrBeginFrame(session, NULL));
			if (start == 0) {
				start = fs.predictedDisplayTime;
			}
			double t = (fs.predictedDisplayTime - start) / 1e9;
			bool closed = t > lead_s + pursuit_s;
			if (t > lead_s + pursuit_s + blink_s) {
				XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
				                      .displayTime = fs.predictedDisplayTime,
				                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE};
				CK(xrEndFrame(session, &fei));
				break;
			}

			// Incommensurate periods sweep the whole range; max about 12 deg/s.
			double u = t < lead_s ? 0.0 : t - lead_s;
			double yaw = closed ? 0.0 : 25.0 * sin(2.0 * M_PI * u / 13.0);
			double pitch = closed ? 0.0 : 15.0 * sin(2.0 * M_PI * u / 8.3);

			// Keeping the gaze space in use keeps the driver processing frames,
			// which is what feeds BIGEYE_CAPTURE. The live gaze is logged too.
			CK(xrSyncActions(session, &asi_r));
			double gy = 0, gp = 0;
			XrSpaceLocation loc = {.type = XR_TYPE_SPACE_LOCATION};
			if (XR_SUCCEEDED(xrLocateSpace(gaze_space, view_space, fs.predictedDisplayTime, &loc)) &&
			    (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)) {
				gaze_angles_from_quat(loc.pose.orientation, &gy, &gp);
			}
			struct timespec mono = {0};
			if (to_timespec != NULL) {
				to_timespec(instance, fs.predictedDisplayTime, &mono);
			}
			long long mono_ns = (long long)mono.tv_sec * 1000000000LL + mono.tv_nsec;
			fprintf(log, "%lld %.3f %.3f %d %.2f %.2f\n", mono_ns, yaw, pitch, closed ? 1 : 0, gy, gp);

			solid_swapchain_present(&vk, &bg_sc);
			solid_swapchain_present(&vk, &dot_sc);

			XrCompositionLayerQuad bg_quad = {
			    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
			    .space = view_space,
			    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
			    .subImage = {.swapchain = bg_sc.swapchain, .imageRect = {.extent = {64, 64}}},
			    .pose = {.orientation = {.w = 1}, .position = {0, 0, -2.0f}},
			    .size = {8.0f, 8.0f},
			};
			XrCompositionLayerQuad dot = {
			    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
			    .space = view_space,
			    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
			    .subImage = {.swapchain = dot_sc.swapchain, .imageRect = {.extent = {16, 16}}},
			    .pose = quad_pose_from_angles(yaw, pitch),
			    .size = {0.015f, 0.015f},
			};
			const XrCompositionLayerBaseHeader *layers[] = {(XrCompositionLayerBaseHeader *)&bg_quad,
			                                                (XrCompositionLayerBaseHeader *)&dot};
			XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
			                      .displayTime = fs.predictedDisplayTime,
			                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
			                      .layerCount = fs.shouldRender ? (closed ? 1 : 2) : 0,
			                      .layers = layers};
			CK(xrEndFrame(session, &fei));
		}
		fclose(log);
		printf("wrote %s\n", record_path);
		xrDestroySession(session);
		xrDestroyInstance(instance);
		return 0;
	}

	/*
	 * Follow mode: a square the size of the foveated full-resolution region,
	 * centred on the reported gaze, with a red dot at its centre and fixed
	 * reference dots at known angles to judge accuracy against.
	 */
	if (follow) {
		const char *fov_env = getenv("BIGEYE_DEMO_FOV_DEG");
		double fov = fov_env != NULL ? atof(fov_env) : 38.0;
		const char *secs_env = getenv("BIGEYE_DEMO_SECONDS");
		double seconds = secs_env != NULL ? atof(secs_env) : 60.0;
		float region = (float)(2.0 * tan(RAD(fov / 2.0)));
		static const struct target refs[] = {{0, 0}, {15, 0}, {-15, 0}, {0, 8}, {0, -8}};
		XrActiveActionSet active_f = {.actionSet = action_set};
		XrActionsSyncInfo asi_f = {.type = XR_TYPE_ACTIONS_SYNC_INFO,
		                           .countActiveActionSets = 1,
		                           .activeActionSets = &active_f};
		XrTime start = 0;
		int frame = 0;
		const char *log_path = getenv("BIGEYE_DEMO_LOG");
		FILE *log = log_path != NULL ? fopen(log_path, "w") : NULL;
		printf("Follow mode: %.0f deg region, %.0f s. Reference dots at 0, +-15 yaw, +-8 pitch.\n", fov,
		       seconds);

		while (true) {
			XrEventDataBuffer ev = {.type = XR_TYPE_EVENT_DATA_BUFFER};
			while (xrPollEvent(instance, &ev) == XR_SUCCESS) {
				ev.type = XR_TYPE_EVENT_DATA_BUFFER;
			}
			XrFrameState fs = {.type = XR_TYPE_FRAME_STATE};
			CK(xrWaitFrame(session, NULL, &fs));
			CK(xrBeginFrame(session, NULL));
			if (start == 0) {
				start = fs.predictedDisplayTime;
			}
			if ((double)(fs.predictedDisplayTime - start) > seconds * 1e9) {
				XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
				                      .displayTime = fs.predictedDisplayTime,
				                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE};
				CK(xrEndFrame(session, &fei));
				break;
			}

			CK(xrSyncActions(session, &asi_f));
			double gy = 0, gp = 0;
			bool tracked = false;
			XrSpaceLocation loc = {.type = XR_TYPE_SPACE_LOCATION};
			if (XR_SUCCEEDED(xrLocateSpace(gaze_space, view_space, fs.predictedDisplayTime, &loc)) &&
			    (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)) {
				gaze_angles_from_quat(loc.pose.orientation, &gy, &gp);
				tracked = true;
			}
			if (log != NULL) {
				fprintf(log, "%.3f %.2f %.2f %d\n", (fs.predictedDisplayTime - start) / 1e9, gy, gp, tracked);
			}
			if (frame++ % 45 == 0) {
				printf("\rgaze yaw %+6.1f pitch %+6.1f %s   ", gy, gp, tracked ? "" : "(not tracked)");
				fflush(stdout);
			}

			solid_swapchain_present(&vk, &bg_sc);
			solid_swapchain_present(&vk, &fg_sc);
			solid_swapchain_present(&vk, &dot_sc);
			solid_swapchain_present(&vk, &ref_sc);

			XrCompositionLayerQuad bg_quad = {
			    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
			    .space = view_space,
			    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
			    .subImage = {.swapchain = bg_sc.swapchain, .imageRect = {.extent = {64, 64}}},
			    .pose = {.orientation = {.w = 1}, .position = {0, 0, -2.0f}},
			    .size = {8.0f, 8.0f},
			};
			XrCompositionLayerQuad region_quad = {
			    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
			    .space = view_space,
			    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
			    .subImage = {.swapchain = fg_sc.swapchain, .imageRect = {.extent = {128, 128}}},
			    .pose = quad_pose_from_angles(gy, gp),
			    .size = {region, region},
			};
			XrCompositionLayerQuad gaze_dot = region_quad;
			gaze_dot.subImage = (XrSwapchainSubImage){.swapchain = dot_sc.swapchain, .imageRect = {.extent = {16, 16}}};
			gaze_dot.size = (XrExtent2Df){0.015f, 0.015f};
			XrCompositionLayerQuad ref_quads[5];
			for (int i = 0; i < 5; i++) {
				ref_quads[i] = gaze_dot;
				ref_quads[i].subImage.swapchain = ref_sc.swapchain;
				ref_quads[i].pose = quad_pose_from_angles(refs[i].yaw_deg, refs[i].pitch_deg);
				// Slightly closer than 1 m so they draw over the region square.
				ref_quads[i].pose.position.x *= 0.99f;
				ref_quads[i].pose.position.y *= 0.99f;
				ref_quads[i].pose.position.z *= 0.99f;
			}
			const XrCompositionLayerBaseHeader *layers[8] = {
			    (XrCompositionLayerBaseHeader *)&bg_quad,     (XrCompositionLayerBaseHeader *)&region_quad,
			    (XrCompositionLayerBaseHeader *)&ref_quads[0], (XrCompositionLayerBaseHeader *)&ref_quads[1],
			    (XrCompositionLayerBaseHeader *)&ref_quads[2], (XrCompositionLayerBaseHeader *)&ref_quads[3],
			    (XrCompositionLayerBaseHeader *)&ref_quads[4], (XrCompositionLayerBaseHeader *)&gaze_dot,
			};
			XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
			                      .displayTime = fs.predictedDisplayTime,
			                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
			                      .layerCount = fs.shouldRender ? 8 : 0,
			                      .layers = layers};
			CK(xrEndFrame(session, &fei));
		}
		printf("\n");
		if (log != NULL) {
			fclose(log);
		}
		xrDestroySession(session);
		xrDestroyInstance(instance);
		return 0;
	}

	/*
	 * Frame loop.
	 */
	XrTime phase_start = 0;
	size_t phase = 0;

	XrActiveActionSet active = {.actionSet = action_set};
	XrActionsSyncInfo asi = {.type = XR_TYPE_ACTIONS_SYNC_INFO,
	                         .countActiveActionSets = 1,
	                         .activeActionSets = &active};

	while (phase < NUM_TARGETS) {
		XrEventDataBuffer ev = {.type = XR_TYPE_EVENT_DATA_BUFFER};
		while (xrPollEvent(instance, &ev) == XR_SUCCESS) {
			ev.type = XR_TYPE_EVENT_DATA_BUFFER;
		}

		XrFrameState fs = {.type = XR_TYPE_FRAME_STATE};
		CK(xrWaitFrame(session, NULL, &fs));
		CK(xrBeginFrame(session, NULL));

		if (phase_start == 0) {
			phase_start = fs.predictedDisplayTime + START_DELAY_NS;
		}
		if (fs.predictedDisplayTime - phase_start > PHASE_NS) {
			phase++;
			phase_start = fs.predictedDisplayTime;
			if (phase >= NUM_TARGETS) {
				XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
				                      .displayTime = fs.predictedDisplayTime,
				                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE};
				CK(xrEndFrame(session, &fei));
				break;
			}
			printf("target %zu/%zu: yaw %+.0f pitch %+.0f\n", phase + 1, NUM_TARGETS,
			       (double)targets[phase].yaw_deg, (double)targets[phase].pitch_deg);
		}

		// Glide from the previous target to the current one over MOVE_NS
		// (smoothstep eased), then hold. Only sample during the hold.
		int64_t elapsed = fs.predictedDisplayTime - phase_start;
		bool holding = elapsed >= MOVE_NS;
		double cur_yaw = targets[phase].yaw_deg;
		double cur_pitch = targets[phase].pitch_deg;
		double show_yaw = cur_yaw, show_pitch = cur_pitch;
		if (!holding && phase > 0) {
			double t = (double)elapsed / (double)MOVE_NS;
			double s = t * t * (3.0 - 2.0 * t); // smoothstep
			show_yaw = targets[phase - 1].yaw_deg + (cur_yaw - targets[phase - 1].yaw_deg) * s;
			show_pitch = targets[phase - 1].pitch_deg + (cur_pitch - targets[phase - 1].pitch_deg) * s;
		}

		CK(xrSyncActions(session, &asi));
		if (holding) {
			XrSpaceLocation loc = {.type = XR_TYPE_SPACE_LOCATION};
			if (XR_SUCCEEDED(xrLocateSpace(gaze_space, view_space, fs.predictedDisplayTime, &loc)) &&
			    (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT)) {
				double gy, gp;
				gaze_angles_from_quat(loc.pose.orientation, &gy, &gp);
				stats[phase].sum_yaw += gy;
				stats[phase].sum_pitch += gp;
				stats[phase].sq_yaw += gy * gy;
				stats[phase].sq_pitch += gp * gp;
				stats[phase].count++;
			}
		}

		solid_swapchain_present(&vk, &bg_sc);
		solid_swapchain_present(&vk, &fg_sc);
		solid_swapchain_present(&vk, &dot_sc);

		// Background: large quad straight ahead filling the view. Foreground:
		// small quad at the target direction, composited on top (later layer).
		XrCompositionLayerQuad bg_quad = {
		    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
		    .space = view_space,
		    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
		    .subImage = {.swapchain = bg_sc.swapchain, .imageRect = {.extent = {64, 64}}},
		    .pose = {.orientation = {.w = 1}, .position = {0, 0, -2.0f}},
		    .size = {8.0f, 8.0f},
		};
		XrCompositionLayerQuad fg_quad = {
		    .type = XR_TYPE_COMPOSITION_LAYER_QUAD,
		    .space = view_space,
		    .eyeVisibility = XR_EYE_VISIBILITY_BOTH,
		    .subImage = {.swapchain = fg_sc.swapchain, .imageRect = {.extent = {128, 128}}},
		    .pose = quad_pose_from_angles(show_yaw, show_pitch),
		    .size = {0.05f, 0.05f},
		};
		XrCompositionLayerQuad dot_quad = fg_quad;
		dot_quad.subImage = (XrSwapchainSubImage){.swapchain = dot_sc.swapchain, .imageRect = {.extent = {16, 16}}};
		dot_quad.size = (XrExtent2Df){0.01f, 0.01f};
		const XrCompositionLayerBaseHeader *layers[] = {(XrCompositionLayerBaseHeader *)&bg_quad,
		                                                (XrCompositionLayerBaseHeader *)&fg_quad,
		                                                (XrCompositionLayerBaseHeader *)&dot_quad};
		XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
		                      .displayTime = fs.predictedDisplayTime,
		                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
		                      .layerCount = fs.shouldRender ? 3 : 0,
		                      .layers = layers};
		CK(xrEndFrame(session, &fei));
	}

	/*
	 * Fit and report.
	 */
	double tx_yaw[NUM_TARGETS], m_yaw[NUM_TARGETS];
	double tx_pitch[NUM_TARGETS], m_pitch[NUM_TARGETS];
	int n = 0;
	printf("\n%8s %8s | %8s %8s | %6s %6s | samples\n", "tgt yaw", "tgt pit", "meas yaw", "meas pit", "sd yaw", "sd pit");
	for (size_t i = 0; i < NUM_TARGETS; i++) {
		if (stats[i].count < 10) {
			printf("%8.1f %8.1f | %17s | %d (skipped)\n", (double)targets[i].yaw_deg,
			       (double)targets[i].pitch_deg, "-", stats[i].count);
			continue;
		}
		tx_yaw[n] = targets[i].yaw_deg;
		m_yaw[n] = stats[i].sum_yaw / stats[i].count;
		tx_pitch[n] = targets[i].pitch_deg;
		m_pitch[n] = stats[i].sum_pitch / stats[i].count;
		double sd_y = sqrt(fmax(stats[i].sq_yaw / stats[i].count - m_yaw[n] * m_yaw[n], 0.0));
		double sd_p = sqrt(fmax(stats[i].sq_pitch / stats[i].count - m_pitch[n] * m_pitch[n], 0.0));
		printf("%8.1f %8.1f | %8.2f %8.2f | %6.2f %6.2f | %d\n", tx_yaw[n], tx_pitch[n], m_yaw[n], m_pitch[n], sd_y,
		       sd_p, stats[i].count);
		n++;
	}

	double cy[POLY_TERMS], cp[POLY_TERMS];
	if (n < 10 || !fit_poly(m_yaw, m_pitch, tx_yaw, n, cy) || !fit_poly(m_yaw, m_pitch, tx_pitch, n, cp)) {
		fprintf(stderr, "not enough valid data to fit\n");
		return 1;
	}

	// Residuals of the fit at the calibration points themselves.
	double max_err = 0;
	printf("\nfit residuals (fitted - target):\n");
	for (int i = 0; i < n; i++) {
		double ey = poly_eval(cy, m_yaw[i], m_pitch[i]) - tx_yaw[i];
		double ep = poly_eval(cp, m_yaw[i], m_pitch[i]) - tx_pitch[i];
		printf("%8.1f %8.1f | yaw %+6.2f  pitch %+6.2f\n", tx_yaw[i], tx_pitch[i], ey, ep);
		max_err = fmax(max_err, fmax(fabs(ey), fabs(ep)));
	}
	printf("max residual %.2f deg\n", max_err);

	// Zero the centre exactly: the least squares fit spreads error over all
	// targets, and the centre is where an offset is most noticeable.
	double off_y = 0, off_p = 0;
	int n_center = 0;
	for (int i = 0; i < n; i++) {
		if (tx_yaw[i] == 0 && tx_pitch[i] == 0) {
			off_y += poly_eval(cy, m_yaw[i], m_pitch[i]);
			off_p += poly_eval(cp, m_yaw[i], m_pitch[i]);
			n_center++;
		}
	}
	if (n_center > 0) {
		off_y /= n_center;
		off_p /= n_center;
	}
	printf("centre offset yaw %+.2f pitch %+.2f\n", off_y, off_p);

	if (getenv("BIGEYE_CALIB_DRY_RUN") != NULL) {
		printf("dry run, not writing calibration\n");
		xrDestroySession(session);
		xrDestroyInstance(instance);
		return 0;
	}

	const char *home = getenv("HOME");
	char dir[512], path[600];
	snprintf(dir, sizeof(dir), "%s/.config/monado", home);
	mkdir(dir, 0755);
	snprintf(path, sizeof(path), "%s/bigeye_calibration.json", dir);
	FILE *f = fopen(path, "w");
	if (f == NULL) {
		fprintf(stderr, "cannot write %s\n", path);
		return 1;
	}
	fprintf(f, "{\n\t\"yaw_poly\": [");
	for (int i = 0; i < POLY_TERMS; i++) {
		fprintf(f, "%s%.6g", i ? ", " : "", cy[i]);
	}
	fprintf(f, "],\n\t\"pitch_poly\": [");
	for (int i = 0; i < POLY_TERMS; i++) {
		fprintf(f, "%s%.6g", i ? ", " : "", cp[i]);
	}
	fprintf(f, "],\n\t\"yaw_offset\": %.3f,\n\t\"pitch_offset\": %.3f\n}\n", off_y, off_p);
	fclose(f);
	printf("wrote %s\n", path);

	xrDestroySession(session);
	xrDestroyInstance(instance);
	return 0;
}
