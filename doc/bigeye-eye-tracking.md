# Bigscreen Beyond 2e eye tracking {#bigeye-eye-tracking}

<!--
Copyright 2026, Samuel Rounce
SPDX-License-Identifier: BSL-1.0
-->

[TOC]

The `bigeye` driver (`d/bigeye`) provides `XR_EXT_eye_gaze_interaction` from
the eye tracking cameras of the Bigscreen Beyond 2e. It runs alongside the
`steamvr_lh` driver, which keeps providing the head pose; the gaze pose is
chained onto it. Tools for calibration and per-user model training live in
`targets/bigeye_calibration`.

## Hardware and capture

The cameras enumerate as one UVC 1.1 device (`35bd:0202`) streaming both eyes
side by side as 800x400 MJPEG at 90 Hz on an isochronous endpoint. The IR
illuminators come on with the stream; no HID command is needed.

Capture uses `d/uvc`, the userspace UVC frameserver over libusb, rather than
V4L2. The camera firmware (v53/v54) reports `dwMaxVideoFrameSize = 0` and, if
any client negotiates a non-MJPG format, latches its probe state until the
headset is power cycled. That breaks uvcvideo buffer allocation and it cannot
recover. With `d/uvc` the driver fills the probe/commit control itself and
sizes its own buffers, so the descriptor value is irrelevant and stock
firmware works. Do not run other capture software (OpenCV's V4L2 backend in
particular) against the device while Monado is using it.

The process needs raw USB access to the device. A udev rule such as

    SUBSYSTEM=="usb", ATTRS{idVendor}=="35bd", ATTRS{idProduct}=="0202", MODE="0660", GROUP="users"

is enough. On NixOS `TAG+="uaccess"` in `services.udev.extraRules` does not
work, because those rules are processed after `73-seat-late.rules`.

## Model and inference

Each eye half is cropped, resized to 128x128 and histogram equalized. The
model input is the last four frames, newest first, two eyes per frame, as
`[1, 8, 128, 128]` in 0..1; the output is `[1, 6]`: pitch, yaw and lid
closedness per eye in 0..1, angles as `(deg + 45) / 90`. The images are
mirrored and the eyes swapped relative to the raw frame by default
(`BIGEYE_FLIP_A`, `BIGEYE_FLIP_B`, `BIGEYE_SWAP_EYES`); this matched the
orientation the reference models expect.

The model file comes from `BIGEYE_EYE_MODEL`; without it the device is not
created. Nothing is bundled. Third-party models such as Project Babble's carry
a non-commercial copyleft license and cannot be shipped with Monado, and on
this hardware their stock model gave usable yaw but almost no vertical
signal. The intended path is a model trained on your own eyes, below.

## Training a per-user model

Two minutes of data are enough. With the service running and
`BIGEYE_CAPTURE=/path/capture.bin` set, run

    bigeye_calibration record /path/labels.txt

Follow the red dot as it drifts around for 120 s, then keep your eyes closed
for the last 15 s after it disappears. The driver writes every preprocessed
frame pair with its `CLOCK_MONOTONIC` timestamp to the capture file while
gaze is in use; the tool logs the target angles with the same clock via
`XR_KHR_convert_timespec_time`. Then

    bigeye_train.py /path/capture.bin /path/labels.txt ~/.local/share/monado/user_model.onnx

trains one small convolutional net per eye from scratch (PyTorch, about a
minute on a GPU, needs `numpy torch onnx onnxscript`) and exports a single
ONNX file in the driver's layout. Point `BIGEYE_EYE_MODEL` at it. Training
errors around 1 degree on both axes are typical.

## Calibration and recentering

Even a per-user model has a small residual mapping error and the headset
never sits the same way twice. Two tools handle this, both writing
`~/.config/monado/bigeye_calibration.json`, which the driver re-reads whenever
the file changes:

- `bigeye_calibration` shows 14 fixation targets (about 35 s), fits a
  bilinear map from reported to true gaze per axis (the cross term handles
  coupling between the axes) and stores the centre residual as an offset. Run
  it after seating the headset. It moves any previous file to `.bak` first
  so it measures the raw mapping.
- `bigeye_calibration recenter` takes 5 s on the centre dot and adds the
  measured residual to the stored offsets. Use it for small drift during a
  session; a large shift of the headset needs a recalibration.

`bigeye_calibration follow` shows a square the size of the foveated
full-resolution region (`BIGEYE_DEMO_FOV_DEG`, default 38) following the
gaze, with fixed reference dots, to judge the result. With a per-user model,
calibration and a recenter, the 14 targets validate within about 2 degrees.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `BIGEYE_EYE_MODEL` | unset | ONNX model path, required |
| `BIGEYE_LOG` | info | log level |
| `BIGEYE_CROP_SIZE`, `BIGEYE_CROP_A_X`, `BIGEYE_CROP_B_X`, `BIGEYE_CROP_Y` | 350, 0, 50, 0 | per-eye crop, offsets into each 400 px half |
| `BIGEYE_FLIP_A`, `BIGEYE_FLIP_B`, `BIGEYE_SWAP_EYES` | true | image orientation fed to the model |
| `BIGEYE_FILTER_FCMIN`, `BIGEYE_FILTER_BETA` | 3.0, 0.02 | one-euro filter |
| `BIGEYE_GAZE_SCALE_DEG` | 45 | angle full scale of the model output |
| `BIGEYE_PITCH_LIMIT_DEG`, `BIGEYE_PITCH_GAIN` | 0 (off), 1 | bound and gain for models with weak vertical response |
| `BIGEYE_CAPTURE` | unset | training capture stream |
| `BIGEYE_DUMP` | unset | write model inputs as a PGM sequence for crop inspection |

All of these are also adjustable at runtime in the debug GUI.

## Limitations

The cameras stream, and the illuminators are lit, for as long as the device
exists; only inference is gated on an application using eye tracking. The
blink handling is a simple weight threshold on the lid outputs. There is no
automatic detection of the headset moving on the face; recenter or
recalibrate when it does.
