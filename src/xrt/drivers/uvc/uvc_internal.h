// Copyright 2017, Philipp Zabel
// Copyright 2019-2021, Jan Schmidt
// Copyright 2025-2026, Beyley Cardellio
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Userspace UVC frameserver implementation
 * @author Philipp Zabel <philipp.zabel@gmail.com>
 * @author Jan Schmidt <jan@centricular.com>
 * @author Beyley Cardellio <ep1cm1n10n123@gmail.com>
 * @ingroup drv_uvc
 */

#pragma once

#include "os/os_threading.h"

#include "util/u_sink.h"
#include "util/u_logging.h"
#include "util/u_var.h"

#include "uvc_interface.h"


/*!
 * Fixed pool of frames handed to consumers. Consumers may hold frames after
 * the stream has stopped, so the pool owns its own lifetime: it is freed by
 * whichever of stream stop or the last frame release comes second.
 */
struct uvc_frame_pool
{
	struct os_mutex lock;
	struct xrt_frame *frames;
	struct xrt_frame **free_frames;
	size_t count;
	size_t num_free;
	//! Set once the stream has stopped and will not take frames again.
	bool retired;
};

struct uvc_fs
{
	struct xrt_fs base;

	struct xrt_frame_node node;

	struct u_sink_debug usd;

	enum u_logging_level log_level;

	struct xrt_fs_mode mode;

	//! Target sink
	struct xrt_frame_sink *sink;

	struct uvc_stream_parameters parameters;

	//! Frame pool for the current stream, NULL when stopped.
	struct uvc_frame_pool *pool;

	//! Frame data destination
	struct xrt_frame *cur_frame;

	//! Total size of a full frame in bytes
	size_t frame_size;
	//! Current frame ID
	int frame_id;
	//! Current PTS being accumulated
	uint32_t cur_pts;
	//! Number of bytes collected from the current frame
	size_t frame_collected;
	//! true if we're skipping the current frame
	bool skip_frame;

	//! Time at which we started skipping frames
	timepoint_ns skip_frame_start;

	//! USB streaming alt_setting
	int alt_setting;

	size_t num_transfers;
	struct libusb_transfer **transfer;
	size_t active_transfers;

	libusb_context *usb_ctx;
	libusb_device_handle *devh;

	bool is_running;

	//! Stream health: isoc packets with a non-OK status, and MJPEG frames
	//! that did not start with a JPEG header. Reported rate limited.
	size_t bad_packets;
	size_t bad_frames;
	size_t good_frames;
	timepoint_ns last_health_report_ns;

	void *get_frame_timestamp_user_data;
	get_frame_timestamp_t get_frame_timestamp;
};
