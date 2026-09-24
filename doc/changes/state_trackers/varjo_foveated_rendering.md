st/oxr: Add XR_VARJO_foveated_rendering. The emulated foveated inset views follow
the system's eye gaze when the app enables it, and
XR_REFERENCE_SPACE_TYPE_COMBINED_EYE_VARJO is exposed for gaze availability
checks. The inset size and the context view size hint are tunable with
OXR_FOVEATED_INSET_FRACTION and OXR_FOVEATED_CONTEXT_SCALE_PERCENTAGE.
