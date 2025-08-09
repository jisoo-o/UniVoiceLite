#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess GRID videos:
- Extract per-frame lip ROI sequence and save as .npy to .../video/output/<basename>.npy
- Extract a face feature from the first frame and save as .pt to .../video/<basename>.pt

Assumes directory layout compatible with your Dataset:
.../<dialect>/<set_type>/<speaker>/video/<...>.mpg
→ saves:
.../<dialect>/<set_type>/<speaker>/video/output/<basename>.npy
.../<dialect>/<set_type>/<speaker>/video/<basename>.pt
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import dlib

# --------------------------
# Utils
# --------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def is_video(name: str):
    name_lower = name.lower()
    return name_lower.endswith(".mpg") 

def bgr2rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# --------------------------
# Lip ROI extraction (per-frame)
# --------------------------
def extract_lip_roi_sequence(video_path, detector, predictor, out_size=(67, 67), fallback="zero"):
    """
    Returns np.array of shape [T, H, W] (grayscale)
    fallback: "zero" | "prev" | "skip"
      - zero: if no face, append zeros
      - prev: if no face, repeat previous ROI (or zeros if first)
      - skip: if no face, skip that frame (variable length)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    lip_rois = []
    prev_roi = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)  # dlib rectangles

        roi_this = None
        for face in faces[:1]:  # first face only
            shape = predictor(gray, face)
            lip_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)], dtype=np.int32)
            x, y, w, h = cv2.boundingRect(lip_pts)
            # safe clamp
            H, W = gray.shape[:2]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 > x0 and y1 > y0:
                roi = gray[y0:y1, x0:x1]
                roi_resized = cv2.resize(roi, out_size, interpolation=cv2.INTER_AREA)
                roi_this = roi_resized
            break

        if roi_this is None:
            if fallback == "skip":
                continue
            elif fallback == "prev" and prev_roi is not None:
                roi_this = prev_roi
            else:
                roi_this = np.zeros(out_size, dtype=np.uint8)

        lip_rois.append(roi_this)
        prev_roi = roi_this

    cap.release()
    return np.stack(lip_rois, axis=0) if len(lip_rois) > 0 else np.zeros((0, out_size[0], out_size[1]), dtype=np.uint8)

# --------------------------
# Face embedding from first frame
# --------------------------
def extract_first_frame_face_feature(video_path, model, preprocess, detect_first_face="dlib"):
    """
    - Reads first frame
    - Detects first face (dlib frontal face)
    - Crops & center-pads to 224x224
    - Returns torch.Tensor [1, D] (D=out_dim)
    """
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame: {video_path}")

    img_rgb = bgr2rgb(frame)
    H, W = img_rgb.shape[:2]
    pil = Image.fromarray(img_rgb)

    # dlib face detect on grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dlib_detector = dlib.get_frontal_face_detector()
    faces = dlib_detector(gray)

    if len(faces) == 0:
        # no face → use full frame center-crop as fallback
        crop = pil
    else:
        face = faces[0]
        x0, y0, x1, y1 = face.left(), face.top(), face.right(), face.bottom()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        crop = pil.crop((x0, y0, x1, y1))

    # letterbox (center-pad) to 224x224
    out_size = (224, 224)
    bg = Image.new("RGB", out_size, "black")
    # keep aspect ratio
    crop.thumbnail(out_size, Image.BILINEAR)
    ox = (out_size[0] - crop.width) // 2
    oy = (out_size[1] - crop.height) // 2
    bg.paste(crop, (ox, oy))

    x = preprocess(bg).unsqueeze(0)  # [1,3,224,224]
    with torch.no_grad():
        feat = model(x)  # [1,D]
    return feat.cpu()

def build_resnet18_head(out_dim=768):

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, out_dim)
    model.eval()
    return model

def build_preprocess():
    return T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

# --------------------------
# Main processing
# --------------------------
def process_tree(base_dir, shape_predictor_path,
                 lip_size=67, lip_fallback="zero",
                 out_dim=768, overwrite=False, verbose=True):
    # dlib models
    if not os.path.exists(shape_predictor_path):
        raise FileNotFoundError(f"Missing shape predictor file: {shape_predictor_path}")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor_path)

    # face feature model
    model = build_resnet18_head(out_dim=out_dim)
    preprocess = build_preprocess()

    for root, _, files in os.walk(base_dir):
        # detect video dir signature (ends with /video or contains video files)
        for fname in files:
            if not is_video(fname):
                continue

            vpath = os.path.join(root, fname)
            base = os.path.splitext(os.path.basename(vpath))[0]

            # expected outputs (align with your Dataset)
            out_dir = os.path.join(root, "output")
            ensure_dir(out_dir)
            npy_path = os.path.join(out_dir, base + ".npy")
            pt_path = os.path.join(root, base + ".pt")

            # skip when already exists
            if not overwrite and os.path.exists(npy_path) and os.path.exists(pt_path):
                if verbose:
                    print(f"[SKIP] both exist: {os.path.relpath(vpath, base_dir)}")
                continue

            if verbose:
                print(f"[PROC] {os.path.relpath(vpath, base_dir)}")

            # 1) lip roi sequence (.npy)
            try:
                if overwrite or not os.path.exists(npy_path):
                    lip_seq = extract_lip_roi_sequence(
                        vpath, detector, predictor,
                        out_size=(lip_size, lip_size),
                        fallback=lip_fallback
                    )
                    # save as uint8 (0-255); consumer can normalize later
                    np.save(npy_path, lip_seq.astype(np.uint8))
                    if verbose:
                        print(f"  - saved lip ROI: {os.path.relpath(npy_path, base_dir)}  shape={lip_seq.shape}")
            except Exception as e:
                print(f"  ! lip roi failed: {e}")

            # 2) face feature first frame (.pt)
            try:
                if overwrite or not os.path.exists(pt_path):
                    feat = extract_first_frame_face_feature(vpath, model, preprocess)  # [1, D]
                    torch.save(feat, pt_path)
                    if verbose:
                        print(f"  - saved face feat: {os.path.relpath(pt_path, base_dir)}  shape={tuple(feat.shape)}")
            except Exception as e:
                print(f"  ! face feature failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess GRID dataset into .npy (lip ROI seq) and .pt (face feature).")
    parser.add_argument("--base_dir", required=True, help="Root directory that contains .../<dialect>/<set_type>/<speaker>/video/*.mpg")
    parser.add_argument("--shape_predictor", required=True, help="Path to dlib's shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--lip_size", type=int, default=67, help="Lip ROI output size (square)")
    parser.add_argument("--lip_fallback", type=str, default="zero", choices=["zero","prev","skip"],
                        help="Fallback when no face on a frame: zero | prev | skip")
    parser.add_argument("--out_dim", type=int, default=768, help="Output feature dim for ResNet18 head (fc)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--quiet", action="store_true", help="Less logs")
    args = parser.parse_args()

    process_tree(
        base_dir=args.base_dir,
        shape_predictor_path=args.shape_predictor,
        lip_size=args.lip_size,
        lip_fallback=args.lip_fallback,
        out_dim=args.out_dim,
        overwrite=args.overwrite,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()
