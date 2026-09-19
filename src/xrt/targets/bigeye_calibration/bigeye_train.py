#!/usr/bin/env python3
# Copyright 2026, Samuel Rounce
# SPDX-License-Identifier: BSL-1.0
"""Train a per-user gaze model for the Monado bigeye driver.

Inputs are a BIGEYE_CAPTURE stream from the driver (int64 timestamp followed
by the two 128x128 preprocessed eye images, per frame) and the label log from
`bigeye_calibration record` (timestamp, target yaw, pitch, eyes-closed flag).
Frames are joined to labels by nearest timestamp, stacked with the three
previous frames the way the driver feeds the model, and a small convolutional
net per eye is trained from scratch. The export matches the driver's layout:
input [1, 8, 128, 128], output [1, 6] = pitch, yaw, closedness per eye in
0..1 with angles as (deg + 45) / 90.

    bigeye_train.py capture.bin labels.txt out.onnx [--epochs N]
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 128
PLANE = SIZE * SIZE
RECORD = 8 + 2 * PLANE
STACK = 4  # frames per eye, newest first
MAX_JOIN_NS = 25_000_000  # 25 ms, under one frame at 90 Hz
ANGLE_SCALE = 45.0


def load_capture(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // RECORD
    raw = raw[: n * RECORD].reshape(n, RECORD)
    ts = raw[:, :8].copy().view(np.int64).reshape(n)
    imgs = raw[:, 8:].reshape(n, 2, SIZE, SIZE)
    return ts, imgs


def load_labels(path):
    rows = np.loadtxt(path, dtype=np.float64, ndmin=2)
    return rows[:, 0].astype(np.int64), rows[:, 1], rows[:, 2], rows[:, 3] > 0.5


def build_samples(ts, imgs, lts, lyaw, lpitch, lclosed, offset_ns):
    if offset_ns is None and (lts.max() < ts.min() or lts.min() > ts.max()):
        # Different clock bases (old label logs used XrTime): align the starts.
        offset_ns = int(ts[0] - lts[0])
        print(f"no timestamp overlap, aligning starts with offset {offset_ns} ns")
    if offset_ns:
        lts = lts + offset_ns
    order = np.argsort(lts)
    lts, lyaw, lpitch, lclosed = lts[order], lyaw[order], lpitch[order], lclosed[order]
    idx = np.searchsorted(lts, ts)
    idx = np.clip(idx, 1, len(lts) - 1)
    left = np.abs(lts[idx - 1] - ts) < np.abs(lts[idx] - ts)
    nearest = np.where(left, idx - 1, idx)
    ok = np.abs(lts[nearest] - ts) <= MAX_JOIN_NS
    # Need STACK consecutive captured frames; the capture is contiguous so use index.
    ok[: STACK - 1] = False
    frames = np.nonzero(ok)[0]
    yaw = lyaw[nearest[frames]]
    pitch = lpitch[nearest[frames]]
    closed = lclosed[nearest[frames]]
    print(f"{len(ts)} frames, {len(lts)} labels, {len(frames)} joined "
          f"({closed.sum()} eyes-closed)")
    return frames, yaw, pitch, closed


class EyeNet(nn.Module):
    """Small conv net on a 4-frame stack of one eye -> pitch, yaw, closed."""

    def __init__(self):
        super().__init__()
        chans = [STACK, 16, 24, 32, 48, 64, 96]
        self.convs = nn.ModuleList(
            nn.Conv2d(chans[i], chans[i + 1], 3, padding=1) for i in range(6))
        self.fc = nn.Linear(chans[-1], 3)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))
            if i < 5:
                x = F.max_pool2d(x, 2)
        x = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.sigmoid(self.fc(x))


class Merged(nn.Module):
    """Driver-facing wrapper: 8 interleaved channels in, 6 outputs."""

    def __init__(self, left, right):
        super().__init__()
        self.left, self.right = left, right

    def forward(self, x):
        # Driver layout: [c0_t, c1_t, c0_t-1, c1_t-1, ...]; c0 is eye 0.
        e0 = x[:, 0::2]
        e1 = x[:, 1::2]
        return torch.cat([self.left(e0), self.right(e1)], dim=1)


def make_stack(imgs, frame, eye):
    # Newest first, matching the driver's ring order.
    return np.stack([imgs[frame - k, eye] for k in range(STACK)])


def augment(x):
    # x: [B, 4, H, W] float 0..1. Small shifts and intensity changes.
    b = x.shape[0]
    shift = torch.randint(-8, 9, (b, 2))
    for i in range(b):
        x[i] = torch.roll(x[i], shifts=(int(shift[i, 0]), int(shift[i, 1])), dims=(1, 2))
    gain = (0.7 + 0.6 * torch.rand(b, 1, 1, 1, device=x.device))
    bias = (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 0.2
    return (x * gain + bias).clamp(0, 1)


def train_eye(imgs, frames, targets, closed, eye, epochs, device):
    net = EyeNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = max(1, len(frames) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * steps_per_epoch)
    y = torch.tensor(targets, dtype=torch.float32)
    open_mask = torch.tensor(~closed, dtype=torch.float32)

    for epoch in range(epochs):
        perm = np.random.permutation(len(frames))
        total = 0.0
        t0 = time.time()
        for s in range(steps_per_epoch):
            sel = perm[s * 64:(s + 1) * 64]
            x = np.stack([make_stack(imgs, frames[i], eye) for i in sel]).astype(np.float32) / 255.0
            x = augment(torch.from_numpy(x).to(device))
            out = net(x)
            yb = y[sel].to(device)
            m = open_mask[sel].to(device)
            # Angles only count while the eyes are open; closedness always.
            loss_angle = ((out[:, :2] - yb[:, :2]) ** 2).mean(1) * m
            loss_lid = (out[:, 2] - yb[:, 2]) ** 2
            loss = (loss_angle.sum() / m.sum().clamp(min=1)) + loss_lid.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"eye {eye} epoch {epoch + 1}/{epochs} loss {total / steps_per_epoch:.5f} "
              f"({time.time() - t0:.1f}s)")
    return net


def evaluate(net, imgs, frames, targets, closed, eye, device):
    net.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(frames), 256):
            sel = frames[i:i + 256]
            x = np.stack([make_stack(imgs, f, eye) for f in sel]).astype(np.float32) / 255.0
            out = net(torch.from_numpy(x).to(device)).cpu().numpy()
            errs.append((out[:, :2] - targets[i:i + 256, :2]) * 2 * ANGLE_SCALE)
    e = np.concatenate(errs)[~closed]
    print(f"eye {eye} train fit: pitch mean abs err {np.abs(e[:, 0]).mean():.2f} deg, "
          f"yaw {np.abs(e[:, 1]).mean():.2f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("labels")
    ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--offset-ns", type=int, default=None,
                    help="add to label timestamps before joining (auto if clocks do not overlap)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    ts, imgs = load_capture(args.capture)
    frames, yaw, pitch, closed = build_samples(ts, imgs, *load_labels(args.labels), args.offset_ns)
    if len(frames) < 500:
        sys.exit("too few joined samples")

    targets = np.stack([(pitch + ANGLE_SCALE) / (2 * ANGLE_SCALE),
                        (yaw + ANGLE_SCALE) / (2 * ANGLE_SCALE),
                        closed.astype(np.float64)], axis=1)

    nets = []
    for eye in (0, 1):
        net = train_eye(imgs, frames, targets, closed, eye, args.epochs, device)
        evaluate(net, imgs, frames, targets, closed, eye, device)
        nets.append(net.cpu().eval())

    merged = Merged(*nets).eval()
    dummy = torch.zeros(1, 2 * STACK, SIZE, SIZE)
    torch.onnx.export(merged, dummy, args.out, input_names=["input"], output_names=["output"],
                      opset_version=17, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    # The exporter may leave weights in a sidecar; fold them into one file.
    import onnx
    import os
    model = onnx.load(args.out, load_external_data=True)
    onnx.save(model, args.out, save_as_external_data=False)
    sidecar = args.out + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
