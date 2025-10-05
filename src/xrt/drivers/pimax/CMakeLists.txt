# Copyright 2024, Coreforge
# SPDX-License-Identifier: BSL-1.0

add_library(
	drv_pimax STATIC
	fixup.c
	pimax.c
	fixup_list.c
	vp2.c
	)
target_link_libraries(
	drv_pimax
	PRIVATE
		xrt-interfaces
		aux_util
		aux_math
		aux_tracking
		xrt-external-cjson
		HIDAPI::hidapi
	)
target_include_directories(drv_pimax PRIVATE ${HIDAPI_INCLUDE_DIRS})
target_include_directories(drv_pimax SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})