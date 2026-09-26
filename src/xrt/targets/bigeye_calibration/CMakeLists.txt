# Copyright 2026, Samuel Rounce
# SPDX-License-Identifier: BSL-1.0

# Standalone OpenXR client through the loader; libmonado only for the record
# mode, which asks the driver to capture its training set.
find_package(OpenXR QUIET)
find_package(Vulkan QUIET)

if(NOT OpenXR_FOUND OR NOT Vulkan_FOUND OR NOT TARGET monado)
	message(STATUS "bigeye_calibration: OpenXR loader, Vulkan or libmonado not available, skipping")
	return()
endif()

add_executable(bigeye_calibration bigeye_calibration.c)

target_link_libraries(
	bigeye_calibration
	PRIVATE
		OpenXR::openxr_loader
		Vulkan::Vulkan
		xrt-external-openxr
		monado
		m
	)

install(TARGETS bigeye_calibration RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(PROGRAMS bigeye_train.py DESTINATION ${CMAKE_INSTALL_BINDIR} RENAME bigeye_train)
