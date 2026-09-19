// Copyright 2026, Samuel Rounce
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ONNX gaze inference for the Bigeye driver.
 * @author Samuel Rounce <samuelrounce@gmail.com>
 * @ingroup drv_bigeye
 */

#pragma once

#include "util/u_logging.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BIGEYE_INPUT_FRAMES 4
#define BIGEYE_INPUT_SIZE 128
#define BIGEYE_INPUT_CHANNELS (BIGEYE_INPUT_FRAMES * 2)
#define BIGEYE_INPUT_FLOATS (BIGEYE_INPUT_CHANNELS * BIGEYE_INPUT_SIZE * BIGEYE_INPUT_SIZE)
#define BIGEYE_OUTPUT_FLOATS 6

struct bigeye_inference;

struct bigeye_inference *
bigeye_inference_create(const char *model_path, enum u_logging_level log_level);

/*!
 * Run gaze inference.
 *
 * @param input  [1, 8, 128, 128] tensor: the last 4 frames newest first, two
 *               128x128 eye images per frame, values normalized to 0..1.
 * @param output 6 floats in 0..1: pitch, yaw, closedness for each eye.
 */
bool
bigeye_inference_run(struct bigeye_inference *inf,
                     const float input[BIGEYE_INPUT_FLOATS],
                     float output[BIGEYE_OUTPUT_FLOATS]);

void
bigeye_inference_destroy(struct bigeye_inference **inf);

#ifdef __cplusplus
}
#endif
