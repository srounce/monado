# Copyright 2026, Samuel Rounce
# SPDX-License-Identifier: BSL-1.0

# Standalone OpenXR client, needs the OpenXR loader (not the Monado runtime).
find_package(OpenXR QUIET)
find_package(Vulkan QUIET)

if(NOT OpenXR_FOUND OR NOT Vulkan_FOUND)
	message(STATUS "bigeye_calibration: OpenXR loader or Vulkan not found, skipping")
	return()
endif()

add_executable(bigeye_calibration bigeye_calibration.c)

target_link_libraries(
	bigeye_calibration
	PRIVATE
		OpenXR::openxr_loader
		Vulkan::Vulkan
		xrt-external-openxr
		m
	)
