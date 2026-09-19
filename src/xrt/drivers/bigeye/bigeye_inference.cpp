// Copyright 2026, Samuel Rounce
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief  ONNX gaze inference for the Bigeye driver.
 * @author Samuel Rounce <samuelrounce@gmail.com>
 * @ingroup drv_bigeye
 */

#include "bigeye_inference.h"

#include "onnx/onnx_wrapper.hpp"

#include <cstring>
#include <memory>
#include <new>

using xrt::auxiliary::onnx::OnnxWrapper;

struct bigeye_inference
{
	std::unique_ptr<OnnxWrapper> wrap;

	float input_data[BIGEYE_INPUT_FLOATS];
	OrtValue *input_tensor = nullptr;
};

extern "C" struct bigeye_inference *
bigeye_inference_create(const char *model_path, enum u_logging_level log_level)
{
	bigeye_inference *inf = new (std::nothrow) bigeye_inference();
	if (inf == nullptr) {
		return nullptr;
	}

	try {
		inf->wrap = std::make_unique<OnnxWrapper>(log_level, model_path, "bigeye");

		const int64_t shape[4] = {1, BIGEYE_INPUT_CHANNELS, BIGEYE_INPUT_SIZE, BIGEYE_INPUT_SIZE};
		ORT_SAFE(*inf->wrap, CreateTensorWithDataAsOrtValue(          //
		                         inf->wrap->meminfo,                  //
		                         inf->input_data,                     //
		                         sizeof(inf->input_data),             //
		                         shape,                               //
		                         4,                                   //
		                         ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, //
		                         &inf->input_tensor));
	} catch (std::exception &e) {
		U_LOG_IFL_E(log_level, "Failed to load gaze model '%s': %s", model_path, e.what());
		delete inf;
		return nullptr;
	}

	return inf;
}

extern "C" bool
bigeye_inference_run(struct bigeye_inference *inf,
                     const float input[BIGEYE_INPUT_FLOATS],
                     float output[BIGEYE_OUTPUT_FLOATS])
{
	std::memcpy(inf->input_data, input, sizeof(inf->input_data));

	const char *input_names[] = {"input"};
	const char *output_names[] = {"output"};
	const OrtValue *inputs[] = {inf->input_tensor};
	OrtValue *output_tensor = nullptr;

	try {
		ORT_SAFE(*inf->wrap, Run(inf->wrap->session, nullptr, //
		                         input_names, inputs, 1,      //
		                         output_names, 1, &output_tensor));

		float *data = nullptr;
		ORT_SAFE(*inf->wrap, GetTensorMutableData(output_tensor, (void **)&data));
		std::memcpy(output, data, BIGEYE_OUTPUT_FLOATS * sizeof(float));
	} catch (std::exception &e) {
		if (output_tensor != nullptr) {
			inf->wrap->api->ReleaseValue(output_tensor);
		}
		return false;
	}

	inf->wrap->api->ReleaseValue(output_tensor);
	return true;
}

extern "C" void
bigeye_inference_destroy(struct bigeye_inference **inf_ptr)
{
	bigeye_inference *inf = *inf_ptr;
	if (inf == nullptr) {
		return;
	}

	if (inf->input_tensor != nullptr && inf->wrap != nullptr) {
		inf->wrap->api->ReleaseValue(inf->input_tensor);
	}

	delete inf;
	*inf_ptr = nullptr;
}
