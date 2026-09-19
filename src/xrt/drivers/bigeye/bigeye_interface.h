// Copyright 2026, Samuel Rounce
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  Interface to the Bigscreen Beyond 2e eye tracking driver.
 * @author Samuel Rounce <samuelrounce@gmail.com>
 * @ingroup drv_bigeye
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

struct xrt_device;

/*!
 * @defgroup drv_bigeye Bigscreen Beyond 2e eye tracking driver
 * @ingroup drv
 *
 * @brief Driver for the "Bigeye" eye tracking cameras of the Bigscreen
 * Beyond 2e, providing @ref XRT_INPUT_GENERIC_EYE_GAZE_POSE.
 */

/*!
 * Create a Bigeye eye tracking device.
 *
 * Gaze poses are returned in the tracking space of @p head by chaining the
 * inferred gaze orientation with the head pose.
 *
 * Returns NULL if the cameras are not present, the model is not configured
 * (BIGEYE_EYE_MODEL) or initialization fails.
 *
 * @ingroup drv_bigeye
 */
struct xrt_device *
bigeye_device_create(struct xrt_device *head);

#ifdef __cplusplus
}
#endif
