// Copyright 2026, Samuel Rounce
// SPDX-License-Identifier: BSL-1.0
//
// In-headset eye gaze calibration for the Monado bigeye driver.
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

// Least squares fit y = a*x + b.
static bool
fit_line(const double *x, const double *y, int n, double *a, double *b)
{
	double sx = 0, sy = 0, sxx = 0, sxy = 0;
	for (int i = 0; i < n; i++) {
		sx += x[i];
		sy += y[i];
		sxx += x[i] * x[i];
		sxy += x[i] * y[i];
	}
	double denom = n * sxx - sx * sx;
	if (fabs(denom) < 1e-9) {
		return false;
	}
	*a = (n * sxy - sx * sy) / denom;
	*b = (sy - *a * sx) / n;
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
main(void)
{
	setvbuf(stdout, NULL, _IONBF, 0);

	/*
	 * Instance, system.
	 */
	const char *exts[] = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME, "XR_EXT_eye_gaze_interaction"};
	XrInstanceCreateInfo ici = {.type = XR_TYPE_INSTANCE_CREATE_INFO,
	                            .applicationInfo = {.applicationName = "bigeye_calib",
	                                                .apiVersion = XR_API_VERSION_1_0},
	                            .enabledExtensionCount = 2,
	                            .enabledExtensionNames = exts};
	XrInstance instance;
	CK(xrCreateInstance(&ici, &instance));

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
	struct solid_swapchain bg_sc, fg_sc;
	solid_swapchain_init(session, &vk, 64, bg_color, &bg_sc);
	solid_swapchain_init(session, &vk, 128, fg_color, &fg_sc);

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

	printf("Follow the dark square with your eyes, keep your head still.\n");

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
				stats[phase].count++;
			}
		}

		solid_swapchain_present(&vk, &bg_sc);
		solid_swapchain_present(&vk, &fg_sc);

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
		const XrCompositionLayerBaseHeader *layers[] = {(XrCompositionLayerBaseHeader *)&bg_quad,
		                                                (XrCompositionLayerBaseHeader *)&fg_quad};
		XrFrameEndInfo fei = {.type = XR_TYPE_FRAME_END_INFO,
		                      .displayTime = fs.predictedDisplayTime,
		                      .environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE,
		                      .layerCount = fs.shouldRender ? 2 : 0,
		                      .layers = layers};
		CK(xrEndFrame(session, &fei));
	}

	/*
	 * Fit and report.
	 */
	double tx_yaw[NUM_TARGETS], m_yaw[NUM_TARGETS];
	double tx_pitch[NUM_TARGETS], m_pitch[NUM_TARGETS];
	int n = 0;
	printf("\n%8s %8s | %8s %8s | samples\n", "tgt yaw", "tgt pit", "meas yaw", "meas pit");
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
		printf("%8.1f %8.1f | %8.2f %8.2f | %d\n", tx_yaw[n], tx_pitch[n], m_yaw[n], m_pitch[n],
		       stats[i].count);
		n++;
	}

	double ay, by, ap, bp;
	if (n < 4 || !fit_line(tx_yaw, m_yaw, n, &ay, &by) || !fit_line(tx_pitch, m_pitch, n, &ap, &bp)) {
		fprintf(stderr, "not enough valid data to fit\n");
		return 1;
	}

	printf("\nfit: measured_yaw   = %+.3f * true + %+.2f\n", ay, by);
	printf("fit: measured_pitch = %+.3f * true + %+.2f\n", ap, bp);
	if (fabs(ay) < 0.05 || fabs(ap) < 0.05) {
		fprintf(stderr, "WARNING: near-zero gain, calibration unreliable\n");
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
	fprintf(f,
	        "{\n\t\"yaw_gain\": %.4f,\n\t\"yaw_bias\": %.3f,\n\t\"pitch_gain\": %.4f,\n\t\"pitch_bias\": %.3f\n}\n",
	        ay, by, ap, bp);
	fclose(f);
	printf("wrote %s\n", path);

	xrDestroySession(session);
	xrDestroyInstance(instance);
	return 0;
}
