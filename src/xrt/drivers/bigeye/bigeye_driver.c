// Copyright 2026, Samuel Rounce
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Bigscreen Beyond 2e eye tracking driver.
 *
 * Captures the combined 800x400 MJPEG stream from the "Bigeye" cameras via
 * the userspace UVC frameserver, runs gaze inference and exposes the result
 * as @ref XRT_INPUT_GENERIC_EYE_GAZE_POSE.
 *
 * The userspace UVC stack is used instead of V4L2 on purpose: the camera
 * firmware (up to at least v54) reports dwMaxVideoFrameSize = 0 and latches
 * its probe state until power cycle if a client negotiates any non-MJPG
 * format, which breaks uvcvideo. Controlling probe/commit directly sidesteps
 * both problems.
 *
 * @author Samuel Rounce <samuelrounce@gmail.com>
 * @ingroup drv_bigeye
 */

#include "xrt/xrt_device.h"
#include "xrt/xrt_defines.h"
#include "xrt/xrt_frame.h"
#include "xrt/xrt_frameserver.h"

#include "os/os_threading.h"

#include "math/m_api.h"
#include "math/m_filter_one_euro.h"
#include "math/m_relation_history.h"
#include "math/m_space.h"

#include "util/u_debug.h"
#include "util/u_device.h"
#include "util/u_file.h"
#include "util/u_json.h"
#include "util/u_logging.h"
#include "util/u_misc.h"
#include "util/u_sink.h"
#include "util/u_trace_marker.h"
#include "util/u_var.h"

#include "uvc/uvc_interface.h"

#include "bigeye_interface.h"
#include "bigeye_inference.h"

#include <libusb.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>


#define BIGEYE_VID 0x35bd
#define BIGEYE_PID 0x0202

#define BIGEYE_FRAME_WIDTH 800
#define BIGEYE_FRAME_HEIGHT 400
// 90 fps
#define BIGEYE_FRAME_INTERVAL_100NS 111111

/*
 * Per-eye crop excluding the illuminator flare column at the inner edge of
 * each camera view; values matched against captures during prototyping.
 */
// Crop X values are offsets into each 400px eye half (A = left, B = right).
#define BIGEYE_CROP_SIZE 350
#define BIGEYE_CROP_A_X 0
#define BIGEYE_CROP_B_X 50
#define BIGEYE_CROP_Y 0

#define BIGEYE_TRACE(d, ...) U_LOG_XDEV_IFL_T(&d->base, d->log_level, __VA_ARGS__)
#define BIGEYE_DEBUG(d, ...) U_LOG_XDEV_IFL_D(&d->base, d->log_level, __VA_ARGS__)
#define BIGEYE_INFO(d, ...) U_LOG_XDEV_IFL_I(&d->base, d->log_level, __VA_ARGS__)
#define BIGEYE_WARN(d, ...) U_LOG_XDEV_IFL_W(&d->base, d->log_level, __VA_ARGS__)
#define BIGEYE_ERROR(d, ...) U_LOG_XDEV_IFL_E(&d->base, d->log_level, __VA_ARGS__)

DEBUG_GET_ONCE_LOG_OPTION(bigeye_log, "BIGEYE_LOG", U_LOGGING_INFO)
DEBUG_GET_ONCE_OPTION(bigeye_model, "BIGEYE_EYE_MODEL", NULL)
DEBUG_GET_ONCE_FLOAT_OPTION(bigeye_scale, "BIGEYE_GAZE_SCALE_DEG", 45.0)
// The stock model gives little vertical signal on this hardware; bounding the
// amplified pitch keeps outliers from throwing the gaze far off.
DEBUG_GET_ONCE_FLOAT_OPTION(bigeye_pitch_limit, "BIGEYE_PITCH_LIMIT_DEG", 15.0)
// Extra gain on calibrated pitch: the model has a dead zone around straight
// ahead so a fit scaled to the extremes compresses the middle.
DEBUG_GET_ONCE_FLOAT_OPTION(bigeye_pitch_gain, "BIGEYE_PITCH_GAIN", 1.0)
// One-euro filter tuning. Small saccades vanish below the low-speed cutoff,
// so this runs more responsive than the eye tracking defaults.
DEBUG_GET_ONCE_FLOAT_OPTION(bigeye_fcmin, "BIGEYE_FILTER_FCMIN", 3.0)
DEBUG_GET_ONCE_FLOAT_OPTION(bigeye_beta, "BIGEYE_FILTER_BETA", 0.02)
DEBUG_GET_ONCE_NUM_OPTION(bigeye_crop_size, "BIGEYE_CROP_SIZE", BIGEYE_CROP_SIZE)
DEBUG_GET_ONCE_NUM_OPTION(bigeye_crop_a_x, "BIGEYE_CROP_A_X", BIGEYE_CROP_A_X)
DEBUG_GET_ONCE_NUM_OPTION(bigeye_crop_b_x, "BIGEYE_CROP_B_X", BIGEYE_CROP_B_X)
DEBUG_GET_ONCE_NUM_OPTION(bigeye_crop_y, "BIGEYE_CROP_Y", BIGEYE_CROP_Y)
DEBUG_GET_ONCE_OPTION(bigeye_dump, "BIGEYE_DUMP", NULL)
// Training capture: every preprocessed frame pair with its timestamp, raw.
DEBUG_GET_ONCE_OPTION(bigeye_capture, "BIGEYE_CAPTURE", NULL)
// Eye image orientation relative to what the model was trained on. The Beyond
// 2e cameras are mirrored versus the reference rig; with the wrong orientation
// one eye's yaw inverts and the lid-weighted average cancels most of it.
DEBUG_GET_ONCE_BOOL_OPTION(bigeye_flip_a, "BIGEYE_FLIP_A", true)
DEBUG_GET_ONCE_BOOL_OPTION(bigeye_flip_b, "BIGEYE_FLIP_B", true)
DEBUG_GET_ONCE_BOOL_OPTION(bigeye_swap, "BIGEYE_SWAP_EYES", true)

enum bigeye_input_index
{
	BIGEYE_INPUT_EYE_GAZE_POSE = 0,
	BIGEYE_INPUT_COUNT,
};

struct bigeye_device
{
	struct xrt_device base;

	enum u_logging_level log_level;

	//! Receives decoded R8G8B8 frames from the sink chain.
	struct xrt_frame_sink sink;

	//! Owns the frameserver and sink chain nodes.
	struct xrt_frame_context xfctx;

	libusb_context *usb_ctx;
	libusb_device_handle *devh;
	struct xrt_fs *xfs;

	struct os_thread_helper usb_thread;

	//! Head device to chain gaze poses onto, not owned.
	struct xrt_device *head;

	struct bigeye_inference *inference;
	struct m_relation_history *history;
	struct m_filter_euro_vec2 gaze_filter;

	//! Ring of preprocessed eye images, [slot][channel], channel 0 is the
	//! right image half, matching the model's expected channel order.
	uint8_t ring[BIGEYE_INPUT_FRAMES][2][BIGEYE_INPUT_SIZE * BIGEYE_INPUT_SIZE];
	int ring_head;
	int ring_count;

	float tensor[BIGEYE_INPUT_FLOATS];

	struct os_mutex mutex;
	//! Requested via xrt_device::begin_feature, protected by mutex.
	bool feature_enabled;

	// Runtime adjustable via u_var until proper calibration exists.
	float gaze_scale_deg;
	float pitch_limit_deg;
	float pitch_gain;
	float pitch_offset_deg;
	float yaw_offset_deg;
	bool flip_pitch;
	bool flip_yaw;

	//! Per-eye square crop into each 400px half, tunable to keep the pupil
	//! in frame across the gaze range. A = left half, B = right half.
	int crop_size;
	int crop_a_x, crop_b_x, crop_y;
	bool flip_a, flip_b, swap_eyes;

	//! When set, dump the two preprocessed 128x128 model inputs as PGM to
	//! this path every few frames, for offline crop inspection.
	const char *dump_path;
	int dump_counter;

	//! Training capture stream, see BIGEYE_CAPTURE.
	FILE *capture;

	//! Per-user fit from the calibration tool: measured = gain * true + bias,
	//! inverted here. Identity when no calibration file is present.
	struct
	{
		bool loaded;
		bool apply;
		bool poly;
		// Session offsets from the recenter routine, applied after the mapping.
		float yaw_offset, pitch_offset;
		// Hot reload: the file is re-read when its mtime changes.
		int64_t mtime;
		int64_t last_check_ns;
		// true = c0 + c1*y + c2*p + c3*y*p, degrees.
		float yaw_poly[4], pitch_poly[4];
		// Older per-axis format.
		float yaw_bias, yaw_gain_neg, yaw_gain_pos;
		float pitch_bias, pitch_gain_neg, pitch_gain_pos;
	} calib;
};

static inline struct bigeye_device *
bigeye_device(struct xrt_device *xdev)
{
	return (struct bigeye_device *)xdev;
}


/*
 *
 * Image preprocessing.
 *
 */

/*!
 * Crop a square region out of an R8G8B8 frame, bilinearly resize it to
 * 128x128 grayscale (red channel) and histogram-equalize it.
 */
static void
preprocess_eye(const struct xrt_frame *xf, int crop_x, int crop_y, int crop_size, bool flip, uint8_t *out)
{
	const float scale = (float)crop_size / BIGEYE_INPUT_SIZE;
	const uint8_t *data = xf->data;
	const size_t stride = xf->stride;

	for (int y = 0; y < BIGEYE_INPUT_SIZE; y++) {
		float fy = (y + 0.5f) * scale - 0.5f;
		int iy = (int)fy;
		float wy = fy - iy;
		int y0 = crop_y + iy;
		int y1 = y0 + 1 < crop_y + crop_size ? y0 + 1 : y0;

		for (int x = 0; x < BIGEYE_INPUT_SIZE; x++) {
			float fx = (x + 0.5f) * scale - 0.5f;
			int ix = (int)fx;
			float wx = fx - ix;
			int x0 = crop_x + ix;
			int x1 = x0 + 1 < crop_x + crop_size ? x0 + 1 : x0;
			if (flip) {
				x0 = 2 * crop_x + crop_size - 1 - x0;
				x1 = 2 * crop_x + crop_size - 1 - x1;
			}

			float p00 = data[y0 * stride + x0 * 3];
			float p01 = data[y0 * stride + x1 * 3];
			float p10 = data[y1 * stride + x0 * 3];
			float p11 = data[y1 * stride + x1 * 3];

			float v = p00 * (1 - wy) * (1 - wx) + //
			          p01 * (1 - wy) * wx +       //
			          p10 * wy * (1 - wx) +       //
			          p11 * wy * wx;

			out[y * BIGEYE_INPUT_SIZE + x] = (uint8_t)v;
		}
	}

	// Histogram equalization, matching the reference preprocessing.
	uint32_t hist[256] = {0};
	const int count = BIGEYE_INPUT_SIZE * BIGEYE_INPUT_SIZE;
	for (int i = 0; i < count; i++) {
		hist[out[i]]++;
	}

	uint32_t cdf = 0;
	uint32_t cdf_min = 0;
	uint8_t lut[256];
	for (int i = 0; i < 256; i++) {
		cdf += hist[i];
		if (cdf_min == 0) {
			cdf_min = cdf;
		}
		if (cdf == cdf_min) {
			lut[i] = 0;
			continue;
		}
		lut[i] = (uint8_t)((255.0f * (cdf - cdf_min)) / (count - cdf_min) + 0.5f);
	}

	for (int i = 0; i < count; i++) {
		out[i] = lut[out[i]];
	}
}


/*
 *
 * Frame sink and inference.
 *
 */

static void
bigeye_process_result(struct bigeye_device *d, const float output[BIGEYE_OUTPUT_FLOATS], timepoint_ns timestamp_ns)
{
	// Raw values are normalized 0..1; lids report closedness.
	float l_pitch = output[0] * 2.0f - 1.0f;
	float l_yaw = output[1] * 2.0f - 1.0f;
	float l_open = 1.0f - output[2];
	float r_pitch = output[3] * 2.0f - 1.0f;
	float r_yaw = output[4] * 2.0f - 1.0f;
	float r_open = 1.0f - output[5];

	float weight = l_open + r_open;
	if (weight < 0.2f) {
		// Both eyes closed, keep the last gaze.
		return;
	}

	float scale_rad = d->gaze_scale_deg * ((float)M_PI / 180.0f);
	float pitch = (l_pitch * l_open + r_pitch * r_open) / weight * scale_rad;
	float yaw = (l_yaw * l_open + r_yaw * r_open) / weight * scale_rad;

	if (d->flip_pitch) {
		pitch = -pitch;
	}
	if (d->flip_yaw) {
		yaw = -yaw;
	}
	pitch += d->pitch_offset_deg * ((float)M_PI / 180.0f);
	yaw += d->yaw_offset_deg * ((float)M_PI / 180.0f);

	if (d->calib.loaded && d->calib.apply && d->calib.poly) {
		const float k = 180.0f / (float)M_PI;
		float y = yaw * k, p = pitch * k;
		float t[4] = {1, y, p, y * p};
		float ny = 0, np = 0;
		for (int i = 0; i < 4; i++) {
			ny += d->calib.yaw_poly[i] * t[i];
			np += d->calib.pitch_poly[i] * t[i];
		}
		yaw = ny / k;
		pitch = np / k;
		yaw -= d->calib.yaw_offset / k;
		pitch -= d->calib.pitch_offset / k;
	} else if (d->calib.loaded && d->calib.apply) {
		pitch -= d->calib.pitch_bias * ((float)M_PI / 180.0f);
		yaw -= d->calib.yaw_bias * ((float)M_PI / 180.0f);
		pitch /= pitch < 0 ? d->calib.pitch_gain_neg : d->calib.pitch_gain_pos;
		yaw /= yaw < 0 ? d->calib.yaw_gain_neg : d->calib.yaw_gain_pos;
	}

	pitch *= d->pitch_gain;
	float limit = d->pitch_limit_deg * ((float)M_PI / 180.0f);
	if (limit > 0) {
		pitch = fminf(fmaxf(pitch, -limit), limit);
	}

	struct xrt_vec2 filtered;
	m_filter_euro_vec2_run(&d->gaze_filter, timestamp_ns, &(struct xrt_vec2){pitch, yaw}, &filtered);

	struct xrt_space_relation relation = XRT_SPACE_RELATION_ZERO;
	math_quat_from_euler_angles(&(struct xrt_vec3){.x = filtered.x, .y = -filtered.y},
	                            &relation.pose.orientation);
	relation.relation_flags = XRT_SPACE_RELATION_POSITION_VALID_BIT | XRT_SPACE_RELATION_POSITION_TRACKED_BIT |
	                          XRT_SPACE_RELATION_ORIENTATION_VALID_BIT |
	                          XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT;

	m_relation_history_push(d->history, &relation, timestamp_ns);
}

static void
bigeye_load_calibration(struct bigeye_device *d)
{
	char path[1024];
	if (u_file_get_path_in_config_dir("bigeye_calibration.json", path, sizeof(path)) < 0) {
		return;
	}

	struct stat st;
	d->calib.mtime = stat(path, &st) == 0 ? (int64_t)st.st_mtime : 0;

	char *content = u_file_read_content_from_path(path, NULL);
	if (content == NULL) {
		BIGEYE_INFO(d, "No calibration file at %s, using raw gaze mapping", path);
		return;
	}

	cJSON *json = cJSON_Parse(content);
	free(content);
	if (json == NULL) {
		BIGEYE_WARN(d, "Failed to parse %s", path);
		return;
	}

	d->calib.yaw_offset = d->calib.pitch_offset = 0;
	u_json_get_float(u_json_get(json, "yaw_offset"), &d->calib.yaw_offset);
	u_json_get_float(u_json_get(json, "pitch_offset"), &d->calib.pitch_offset);

	bool ok;
	if (u_json_get(json, "yaw_poly") != NULL) {
		ok = u_json_get_float_array(u_json_get(json, "yaw_poly"), d->calib.yaw_poly, 4) == 4 &&
		     u_json_get_float_array(u_json_get(json, "pitch_poly"), d->calib.pitch_poly, 4) == 4;
		cJSON_Delete(json);
		if (!ok) {
			BIGEYE_WARN(d, "Invalid polynomial calibration in %s, ignoring", path);
			return;
		}
		d->calib.poly = true;
		d->calib.loaded = true;
		d->calib.apply = true;
		BIGEYE_INFO(d, "Loaded quadratic calibration from %s (offset yaw %+.2f pitch %+.2f)", path,
		            d->calib.yaw_offset, d->calib.pitch_offset);
		return;
	}
	ok = u_json_get_float(u_json_get(json, "yaw_bias"), &d->calib.yaw_bias) &&
	     u_json_get_float(u_json_get(json, "pitch_bias"), &d->calib.pitch_bias);
	// Piecewise gains per sign; a single-gain file (older format) applies to both.
	if (ok && u_json_get(json, "yaw_gain_neg") != NULL) {
		ok = u_json_get_float(u_json_get(json, "yaw_gain_neg"), &d->calib.yaw_gain_neg) &&
		     u_json_get_float(u_json_get(json, "yaw_gain_pos"), &d->calib.yaw_gain_pos) &&
		     u_json_get_float(u_json_get(json, "pitch_gain_neg"), &d->calib.pitch_gain_neg) &&
		     u_json_get_float(u_json_get(json, "pitch_gain_pos"), &d->calib.pitch_gain_pos);
	} else if (ok) {
		ok = u_json_get_float(u_json_get(json, "yaw_gain"), &d->calib.yaw_gain_neg) &&
		     u_json_get_float(u_json_get(json, "pitch_gain"), &d->calib.pitch_gain_neg);
		d->calib.yaw_gain_pos = d->calib.yaw_gain_neg;
		d->calib.pitch_gain_pos = d->calib.pitch_gain_neg;
	}
	cJSON_Delete(json);

	if (!ok || fabsf(d->calib.yaw_gain_neg) < 0.01f || fabsf(d->calib.yaw_gain_pos) < 0.01f ||
	    fabsf(d->calib.pitch_gain_neg) < 0.01f || fabsf(d->calib.pitch_gain_pos) < 0.01f) {
		BIGEYE_WARN(d, "Invalid calibration in %s, ignoring", path);
		return;
	}

	d->calib.loaded = true;
	d->calib.apply = true;
	BIGEYE_INFO(d, "Loaded calibration: yaw bias %+.2f gain %.3f/%.3f, pitch bias %+.2f gain %.3f/%.3f",
	            d->calib.yaw_bias, d->calib.yaw_gain_neg, d->calib.yaw_gain_pos, d->calib.pitch_bias,
	            d->calib.pitch_gain_neg, d->calib.pitch_gain_pos);
}

// 3 frames/s for 40 s, enough to cover a full calibration run.
#define BIGEYE_DUMP_INTERVAL 30
#define BIGEYE_DUMP_SLOTS 120

// Write the newest two preprocessed model inputs side by side as a single
// 256x128 PGM, for offline inspection of the crop framing. Cycles through
// <dump_path>.NNN.pgm so a run leaves a sequence spanning the gaze range.
static void
bigeye_dump_inputs(struct bigeye_device *d)
{
	char path[1088];
	int slot = (d->dump_counter / BIGEYE_DUMP_INTERVAL) % BIGEYE_DUMP_SLOTS;
	snprintf(path, sizeof(path), "%s.%03d.pgm", d->dump_path, slot);

	FILE *f = fopen(path, "wb");
	if (f == NULL) {
		return;
	}
	const int S = BIGEYE_INPUT_SIZE;
	fprintf(f, "P5\n%d %d\n255\n", S * 2, S);
	const uint8_t *right = d->ring[d->ring_head][0];
	const uint8_t *left = d->ring[d->ring_head][1];
	for (int y = 0; y < S; y++) {
		fwrite(left + y * S, 1, S, f);   // left eye on the left
		fwrite(right + y * S, 1, S, f);  // right eye on the right
	}
	fclose(f);
}

static void
bigeye_sink_push_frame(struct xrt_frame_sink *xfs, struct xrt_frame *xf)
{
	struct bigeye_device *d = container_of(xfs, struct bigeye_device, sink);

	if (xf->format != XRT_FORMAT_R8G8B8 || xf->width != BIGEYE_FRAME_WIDTH ||
	    xf->height != BIGEYE_FRAME_HEIGHT) {
		BIGEYE_WARN(d, "Unexpected frame %ux%u format %u", xf->width, xf->height, xf->format);
		return;
	}

	os_mutex_lock(&d->mutex);
	bool enabled = d->feature_enabled;
	os_mutex_unlock(&d->mutex);

	// Pick up a rewritten calibration (recenter, recalibration) without restart.
	if (xf->timestamp - d->calib.last_check_ns > U_TIME_1S_IN_NS) {
		d->calib.last_check_ns = xf->timestamp;
		char path[1024];
		struct stat st;
		if (u_file_get_path_in_config_dir("bigeye_calibration.json", path, sizeof(path)) >= 0 &&
		    stat(path, &st) == 0 && (int64_t)st.st_mtime != d->calib.mtime) {
			bigeye_load_calibration(d);
		}
	}

	if (!enabled) {
		d->ring_count = 0;
		return;
	}

	d->ring_head = (d->ring_head + 1) % BIGEYE_INPUT_FRAMES;
	// Channel 0 is the right image half (B), channel 1 the left half (A).
	int ch_b = d->swap_eyes ? 1 : 0;
	preprocess_eye(xf, BIGEYE_FRAME_WIDTH / 2 + d->crop_b_x, d->crop_y, d->crop_size, d->flip_b,
	               d->ring[d->ring_head][ch_b]);
	preprocess_eye(xf, d->crop_a_x, d->crop_y, d->crop_size, d->flip_a, d->ring[d->ring_head][1 - ch_b]);

	if (d->dump_path != NULL && ++d->dump_counter % BIGEYE_DUMP_INTERVAL == 0) {
		bigeye_dump_inputs(d);
	}

	if (d->capture != NULL) {
		int64_t ts = xf->timestamp;
		fwrite(&ts, sizeof(ts), 1, d->capture);
		fwrite(d->ring[d->ring_head][0], 1, BIGEYE_INPUT_SIZE * BIGEYE_INPUT_SIZE, d->capture);
		fwrite(d->ring[d->ring_head][1], 1, BIGEYE_INPUT_SIZE * BIGEYE_INPUT_SIZE, d->capture);
	}

	if (d->ring_count < BIGEYE_INPUT_FRAMES) {
		d->ring_count++;
		if (d->ring_count < BIGEYE_INPUT_FRAMES) {
			return;
		}
	}

	// Newest frame first.
	static const float inv = 1.0f / 255.0f;
	const int plane = BIGEYE_INPUT_SIZE * BIGEYE_INPUT_SIZE;
	for (int i = 0; i < BIGEYE_INPUT_FRAMES; i++) {
		int slot = (d->ring_head - i + BIGEYE_INPUT_FRAMES) % BIGEYE_INPUT_FRAMES;
		for (int c = 0; c < 2; c++) {
			const uint8_t *src = d->ring[slot][c];
			float *dst = d->tensor + (i * 2 + c) * plane;
			for (int p = 0; p < plane; p++) {
				dst[p] = src[p] * inv;
			}
		}
	}

	float output[BIGEYE_OUTPUT_FLOATS];
	if (!bigeye_inference_run(d->inference, d->tensor, output)) {
		BIGEYE_WARN(d, "Gaze inference failed");
		return;
	}

	bigeye_process_result(d, output, xf->timestamp);
}


/*
 *
 * USB plumbing.
 *
 */

static bool
bigeye_setup_stream_parameters(uint16_t vid,
                               uint16_t pid,
                               bool is_usb2,
                               libusb_device_handle *devh,
                               struct uvc_probe_commit_control *control,
                               struct uvc_stream_parameters *parameters,
                               size_t *packet_size,
                               int *alt_setting,
                               void *user_data)
{
	if (vid != BIGEYE_VID || pid != BIGEYE_PID) {
		return false;
	}

	control->dwFrameInterval = __cpu_to_le32(BIGEYE_FRAME_INTERVAL_100NS);
	/*
	 * The firmware reports dwMaxVideoFrameSize = 0 in its descriptors and,
	 * when wedged, in probe responses; the value here only sizes our own
	 * buffers so use the sane descriptor value (800 * 400 * 2).
	 */
	control->dwMaxVideoFrameSize = __cpu_to_le32(640000);
	control->dwMaxPayloadTransferSize = __cpu_to_le32(1024);

	parameters->format = XRT_FORMAT_MJPEG;
	parameters->width = BIGEYE_FRAME_WIDTH;
	parameters->height = BIGEYE_FRAME_HEIGHT;
	parameters->stride = BIGEYE_FRAME_WIDTH;
	parameters->endpoint_address = 0x82;
	// The Bigeye is a UVC 1.1 device and answers probes with the 34 byte control.
	parameters->probe_commit_size = sizeof(struct uvc_probe_commit_control_1_1);

	*packet_size = 1024;
	*alt_setting = 1;

	return true;
}

static void *
bigeye_usb_thread(void *ptr)
{
	struct bigeye_device *d = ptr;

	U_TRACE_SET_THREAD_NAME("Bigeye USB");
	os_thread_helper_name(&d->usb_thread, "Bigeye USB");

	os_thread_helper_lock(&d->usb_thread);
	while (os_thread_helper_is_running_locked(&d->usb_thread)) {
		os_thread_helper_unlock(&d->usb_thread);

		int ret = libusb_handle_events_timeout_completed(
		    d->usb_ctx, &(struct timeval){.tv_sec = 0, .tv_usec = 100000}, NULL);
		if (ret < 0 && ret != LIBUSB_ERROR_TIMEOUT) {
			BIGEYE_ERROR(d, "libusb event handling failed: %d", ret);
			return NULL;
		}

		os_thread_helper_lock(&d->usb_thread);
	}
	os_thread_helper_unlock(&d->usb_thread);

	return NULL;
}


/*
 *
 * Device functions.
 *
 */

static xrt_result_t
bigeye_get_tracked_pose(struct xrt_device *xdev,
                        enum xrt_input_name name,
                        int64_t at_timestamp_ns,
                        struct xrt_space_relation *out_relation)
{
	struct bigeye_device *d = bigeye_device(xdev);

	if (name != XRT_INPUT_GENERIC_EYE_GAZE_POSE) {
		BIGEYE_ERROR(d, "Unknown input name");
		return XRT_ERROR_INPUT_UNSUPPORTED;
	}

	struct xrt_relation_chain chain = {0};

	// Gaze relative to the head, then the head in its tracking space.
	m_relation_history_get(d->history, at_timestamp_ns, m_relation_chain_reserve(&chain));

	struct xrt_space_relation head_relation = XRT_SPACE_RELATION_ZERO;
	xrt_device_get_tracked_pose(d->head, XRT_INPUT_GENERIC_HEAD_POSE, at_timestamp_ns, &head_relation);
	m_relation_chain_push_relation(&chain, &head_relation);

	m_relation_chain_resolve(&chain, out_relation);

	return XRT_SUCCESS;
}

static xrt_result_t
bigeye_begin_feature(struct xrt_device *xdev, enum xrt_device_feature_type type)
{
	struct bigeye_device *d = bigeye_device(xdev);

	if (type != XRT_DEVICE_FEATURE_EYE_TRACKING) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}

	os_mutex_lock(&d->mutex);
	d->feature_enabled = true;
	os_mutex_unlock(&d->mutex);

	BIGEYE_DEBUG(d, "Eye tracking enabled");

	return XRT_SUCCESS;
}

static xrt_result_t
bigeye_end_feature(struct xrt_device *xdev, enum xrt_device_feature_type type)
{
	struct bigeye_device *d = bigeye_device(xdev);

	if (type != XRT_DEVICE_FEATURE_EYE_TRACKING) {
		return XRT_ERROR_FEATURE_NOT_SUPPORTED;
	}

	os_mutex_lock(&d->mutex);
	d->feature_enabled = false;
	os_mutex_unlock(&d->mutex);

	BIGEYE_DEBUG(d, "Eye tracking disabled");

	return XRT_SUCCESS;
}

static void
bigeye_destroy(struct xrt_device *xdev)
{
	struct bigeye_device *d = bigeye_device(xdev);

	u_var_remove_root(d);

	// Stops streaming and destroys the frameserver.
	xrt_frame_context_destroy_nodes(&d->xfctx);

	os_thread_helper_destroy(&d->usb_thread);

	if (d->devh != NULL) {
		libusb_close(d->devh);
	}
	if (d->usb_ctx != NULL) {
		libusb_exit(d->usb_ctx);
	}

	if (d->inference != NULL) {
		bigeye_inference_destroy(&d->inference);
	}
	if (d->capture != NULL) {
		fclose(d->capture);
	}
	if (d->history != NULL) {
		m_relation_history_destroy(&d->history);
	}

	os_mutex_destroy(&d->mutex);

	u_device_free(&d->base);
}


/*
 *
 * Bindings.
 *
 */

static struct xrt_binding_input_pair eye_gaze_inputs[] = {
    {XRT_INPUT_GENERIC_EYE_GAZE_POSE, XRT_INPUT_GENERIC_EYE_GAZE_POSE},
};

static struct xrt_binding_profile binding_profiles[] = {
    {
        .name = XRT_DEVICE_EYE_GAZE_INTERACTION,
        .inputs = eye_gaze_inputs,
        .input_count = ARRAY_SIZE(eye_gaze_inputs),
        .outputs = NULL,
        .output_count = 0,
    },
};


/*
 *
 * 'Exported' functions.
 *
 */

struct xrt_device *
bigeye_device_create(struct xrt_device *head)
{
	enum u_logging_level log_level = debug_get_log_option_bigeye_log();

	const char *model_path = debug_get_option_bigeye_model();
	if (model_path == NULL) {
		U_LOG_IFL_I(log_level, "BIGEYE_EYE_MODEL not set, not creating Bigeye eye tracking device");
		return NULL;
	}

	libusb_context *usb_ctx = NULL;
	int ret = libusb_init(&usb_ctx);
	if (ret < 0) {
		U_LOG_IFL_E(log_level, "Failed to init libusb: %d", ret);
		return NULL;
	}

	libusb_device_handle *devh = libusb_open_device_with_vid_pid(usb_ctx, BIGEYE_VID, BIGEYE_PID);
	if (devh == NULL) {
		U_LOG_IFL_I(log_level, "No Bigeye cameras found (or no permission to open them)");
		libusb_exit(usb_ctx);
		return NULL;
	}

	struct bigeye_device *d = U_DEVICE_ALLOCATE(struct bigeye_device, U_DEVICE_ALLOC_NO_FLAGS, //
	                                            BIGEYE_INPUT_COUNT, 0);
	d->log_level = log_level;
	d->usb_ctx = usb_ctx;
	d->devh = devh;
	d->head = head;
	d->gaze_scale_deg = (float)debug_get_float_option_bigeye_scale();
	d->pitch_limit_deg = (float)debug_get_float_option_bigeye_pitch_limit();
	d->pitch_gain = (float)debug_get_float_option_bigeye_pitch_gain();
	d->crop_size = (int)debug_get_num_option_bigeye_crop_size();
	d->crop_a_x = (int)debug_get_num_option_bigeye_crop_a_x();
	d->crop_b_x = (int)debug_get_num_option_bigeye_crop_b_x();
	d->crop_y = (int)debug_get_num_option_bigeye_crop_y();
	d->dump_path = debug_get_option_bigeye_dump();
	if (debug_get_option_bigeye_capture() != NULL) {
		d->capture = fopen(debug_get_option_bigeye_capture(), "wb");
		if (d->capture == NULL) {
			U_LOG_IFL_W(log_level, "Cannot open capture file %s", debug_get_option_bigeye_capture());
		}
	}
	d->flip_a = debug_get_bool_option_bigeye_flip_a();
	d->flip_b = debug_get_bool_option_bigeye_flip_b();
	d->swap_eyes = debug_get_bool_option_bigeye_swap();
	d->sink.push_frame = bigeye_sink_push_frame;

	os_mutex_init(&d->mutex);
	os_thread_helper_init(&d->usb_thread);
	m_relation_history_create(&d->history);
	m_filter_euro_vec2_init(&d->gaze_filter, debug_get_float_option_bigeye_fcmin(),
	                        M_EURO_FILTER_EYE_TRACKING_FCMIN_D, debug_get_float_option_bigeye_beta());

	u_var_add_root(d, "Bigeye eye tracker", true);
	u_var_add_log_level(d, &d->log_level, "Log level");
	u_var_add_f32(d, &d->gaze_scale_deg, "Gaze scale (deg)");
	u_var_add_f32(d, &d->pitch_limit_deg, "Pitch limit (deg, 0=off)");
	u_var_add_f32(d, &d->pitch_gain, "Pitch gain");
	u_var_add_f32(d, &d->pitch_offset_deg, "Pitch offset (deg)");
	u_var_add_f32(d, &d->yaw_offset_deg, "Yaw offset (deg)");
	u_var_add_bool(d, &d->flip_pitch, "Flip pitch");
	u_var_add_bool(d, &d->flip_yaw, "Flip yaw");
	u_var_add_bool(d, &d->calib.apply, "Apply calibration");
	u_var_add_i32(d, &d->crop_size, "Crop size (px)");
	u_var_add_i32(d, &d->crop_a_x, "Crop left X (px)");
	u_var_add_i32(d, &d->crop_b_x, "Crop right X (px)");
	u_var_add_i32(d, &d->crop_y, "Crop Y (px)");
	u_var_add_bool(d, &d->flip_a, "Flip left image");
	u_var_add_bool(d, &d->flip_b, "Flip right image");
	u_var_add_bool(d, &d->swap_eyes, "Swap eyes");

	bigeye_load_calibration(d);

	d->inference = bigeye_inference_create(model_path, log_level);
	if (d->inference == NULL) {
		U_LOG_IFL_E(log_level, "Failed to create gaze inference from '%s'", model_path);
		goto error;
	}

	// USB event thread must run before streaming starts.
	ret = os_thread_helper_start(&d->usb_thread, bigeye_usb_thread, d);
	if (ret != 0) {
		BIGEYE_ERROR(d, "Failed to start USB thread");
		goto error;
	}

	struct libusb_device_descriptor desc;
	libusb_get_device_descriptor(libusb_get_device(devh), &desc);

	ret = uvc_fs_create(usb_ctx, devh, &desc, bigeye_setup_stream_parameters, NULL, d, &d->xfctx, &d->xfs);
	if (ret < 0) {
		BIGEYE_ERROR(d, "Failed to create UVC frameserver: %d", ret);
		goto error;
	}

	// fs -> queue -> MJPEG decode -> our sink.
	struct xrt_frame_sink *converter = NULL;
	u_sink_create_to_r8g8b8_or_l8(&d->xfctx, &d->sink, &converter);
	struct xrt_frame_sink *queue = NULL;
	u_sink_queue_create(&d->xfctx, 2, converter, &queue);

	if (!xrt_fs_stream_start(d->xfs, queue, XRT_FS_CAPTURE_TYPE_TRACKING, 0)) {
		BIGEYE_ERROR(d, "Failed to start camera stream");
		goto error;
	}

	// Device setup.
	d->base.name = XRT_DEVICE_EYE_GAZE_INTERACTION;
	d->base.device_type = XRT_DEVICE_TYPE_EYE_TRACKER;
	snprintf(d->base.str, XRT_DEVICE_NAME_LEN, "Bigscreen Beyond 2e Eye Tracker");
	snprintf(d->base.serial, XRT_DEVICE_NAME_LEN, "Bigscreen Beyond 2e Eye Tracker");

	d->base.inputs[BIGEYE_INPUT_EYE_GAZE_POSE].name = XRT_INPUT_GENERIC_EYE_GAZE_POSE;
	d->base.binding_profiles = binding_profiles;
	d->base.binding_profile_count = ARRAY_SIZE(binding_profiles);

	d->base.tracking_origin = head->tracking_origin;
	d->base.supported.eye_gaze = true;

	u_device_populate_function_pointers(&d->base, bigeye_get_tracked_pose, bigeye_destroy);
	d->base.begin_feature = bigeye_begin_feature;
	d->base.end_feature = bigeye_end_feature;

	BIGEYE_INFO(d, "Created Bigeye eye tracking device, model '%s'", model_path);

	return &d->base;

error:
	bigeye_destroy(&d->base);
	return NULL;
}
