#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
All-in-One Evaluation Pipeline — Multi-Subject Edition
================================================================================
รองรับ 2 รูปแบบ Dataset อัตโนมัติ (auto-detected):

  [Single-subject]  --dir clips_mp4/0
                    └── <exercise>/cam*.mp4

  [Multi-subject]   --dir clips_mp4_cam1
                    └── <subject>/<exercise>/cam1.mp4

โมเดลที่ใช้ประเมิน:
  - YOLOv8-pose     (PyTorch MPS / CUDA / CPU)
  - MoveNet Thunder (LiteRT / TFLite — XNNPACK)
  - MediaPipe Pose Landmarker Heavy

ออก output:
  - per-video CSV / plot
  - batch_evaluation_summary.csv  (+ subject_id column)
  - model_performance_summary.csv, mae_by_exercise.csv, eda_analysis_plots.png
================================================================================
"""

import os
import sys
import argparse
import glob
import time
import requests
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless-safe (ใช้งานได้บน server/RunPod ที่ไม่มี display)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

import torch
import mediapipe as mp
from ultralytics import YOLO
from ai_edge_litert.interpreter import Interpreter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ------------------------------------------------------------------------------
# Exercise Metadata  (UCO Rehabilitation Dataset — 16 exercises)
# ------------------------------------------------------------------------------
EXERCISE_INFO = {
    1:  ("Bending the knee without support while sitting", "Seated",   "left",  "lower"),
    2:  ("Bending the knee with support while sitting",    "Seated",   "left",  "lower"),
    3:  ("Lift the extended leg",                          "Supine",   "left",  "lower"),
    4:  ("Bending the knee with bed support",              "Supine",   "left",  "lower"),
    5:  ("Bending the knee without support while sitting", "Seated",   "right", "lower"),
    6:  ("Bending the knee with support while sitting",    "Seated",   "right", "lower"),
    7:  ("Lift the extended leg",                          "Supine",   "right", "lower"),
    8:  ("Bending the knee with bed support",              "Supine",   "right", "lower"),
    9:  ("Shoulder flexion",                               "Seated",   "left",  "upper"),
    10: ("Horizontal weighted openings",                   "Standing", "left",  "upper"),
    11: ("External rotation of shoulders with elastic band","Standing","left",  "upper"),
    12: ("Circular pendulum",                              "Standing", "left",  "upper"),
    13: ("Shoulder flexion",                               "Seated",   "right", "upper"),
    14: ("Horizontal weighted openings",                   "Standing", "right", "upper"),
    15: ("External rotation of shoulders with elastic band","Standing","right", "upper"),
    16: ("Circular pendulum",                              "Standing", "right", "upper"),
}

ROTATED_EXERCISE_IDS = {3, 4, 7, 8}   # Supine poses → rotate 90° CW before inference

MOVENET_URL     = "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite"
MEDIAPIPE_URL   = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"


# ==============================================================================
# Utility Functions  (ไม่มีการเปลี่ยนแปลง logic — เหมือนเดิมทุกบรรทัด)
# ==============================================================================
def get_exercise_info(exercise_id):
    if exercise_id not in EXERCISE_INFO:
        raise ValueError(f"Exercise ID {exercise_id} is not in EXERCISE_INFO (expected 1-16).")
    return EXERCISE_INFO[exercise_id]


def ensure_model_files(models_dir="models"):
    """Download MoveNet + MediaPipe model files if not already present."""
    os.makedirs(models_dir, exist_ok=True)
    movenet_path   = os.path.join(models_dir, "movenet_thunder.tflite")
    mediapipe_path = os.path.join(models_dir, "pose_landmarker_heavy.task")

    if not os.path.exists(movenet_path):
        print("[DOWNLOAD] Downloading MoveNet Thunder TFLite...")
        r = requests.get(MOVENET_URL, allow_redirects=True)
        with open(movenet_path, "wb") as f:
            f.write(r.content)
        print(f"[DOWNLOAD] MoveNet saved → {movenet_path} ({os.path.getsize(movenet_path)/1e6:.1f} MB)")

    if not os.path.exists(mediapipe_path):
        print("[DOWNLOAD] Downloading MediaPipe Pose Landmarker Heavy...")
        r = requests.get(MEDIAPIPE_URL, allow_redirects=True)
        with open(mediapipe_path, "wb") as f:
            f.write(r.content)
        print(f"[DOWNLOAD] MediaPipe saved → {mediapipe_path} ({os.path.getsize(mediapipe_path)/1e6:.1f} MB)")

    return movenet_path, mediapipe_path


def detect_device(force=None):
    """Return best available PyTorch device string."""
    if force:
        print(f"[DEVICE] Forced: {force.upper()}")
        return force
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[DEVICE] NVIDIA CUDA GPU: {name} → YOLOv8 will use CUDA.")
        return "0"          # CUDA device index as string
    if torch.backends.mps.is_available():
        print("[DEVICE] Apple Silicon MPS available → YOLOv8 will use MPS.")
        return "mps"
    print("[DEVICE] No GPU found → CPU mode.")
    return "cpu"


def calculate_angle(a, b, c):
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    ba, bc  = a - b, c - b
    n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if n_ba < 1e-6 or n_bc < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0))))


def compute_spatial_error(pt_pred, pt_gt):
    if pt_pred is None or pt_gt is None:
        return float("nan")
    if any(np.isnan(pt_pred)) or any(np.isnan(pt_gt)):
        return float("nan")
    if (pt_pred[0] == 0 and pt_pred[1] == 0) or (pt_gt[0] == 0 and pt_gt[1] == 0):
        return float("nan")
    return float(np.linalg.norm(np.array(pt_pred[:2]) - np.array(pt_gt[:2])))


def map_rotated_kps_to_orig(x, y, orig_w, orig_h, rotation):
    if rotation == "90_cw":
        return y, orig_h - x
    elif rotation == "90_ccw":
        return orig_w - y, x
    return x, y


# ==============================================================================
# Dataset Discovery  ← 3 LINES CHANGED HERE (the only structural change)
# ==============================================================================
def discover_pairs(root_dir):
    """
    Auto-detect dataset format and return list of (video_path, gt_path, subject_id, exercise_dir).

    Format A — Single-subject (old):  root/<exercise>/cam*.mp4
    Format B — Multi-subject (new):   root/<subject>/<exercise>/cam1.mp4
    """
    pairs        = []
    skipped_no_gt = []

    first_level = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
    ])

    if not first_level:
        print("[WARN] No subdirectories found under root_dir.")
        return pairs, skipped_no_gt, "unknown"

    # Detect format: does first subdir contain video files directly?
    probe_path   = os.path.join(root_dir, first_level[0])
    has_videos   = bool(glob.glob(os.path.join(probe_path, "cam*.mp4")))

    if has_videos:
        # ─── Format A: single-subject ───────────────────────────────────────
        fmt = "single-subject"
        subject_id = os.path.basename(root_dir)   # use folder name as subject label
        for exercise_dir in first_level:
            ex_path = os.path.join(root_dir, exercise_dir)
            for v in sorted(glob.glob(os.path.join(ex_path, "cam*.mp4"))):
                gt = v.replace(".mp4", "_p2d.txt")
                entry = (v, gt, subject_id, exercise_dir)
                if os.path.exists(gt):
                    pairs.append(entry)
                else:
                    skipped_no_gt.append(entry)
    else:
        # ─── Format B: multi-subject ────────────────────────────────────────
        fmt = "multi-subject"
        for subject_dir in first_level:
            subject_path = os.path.join(root_dir, subject_dir)
            if not os.path.isdir(subject_path):
                continue
            exercise_dirs = sorted([
                d for d in os.listdir(subject_path)
                if os.path.isdir(os.path.join(subject_path, d)) and not d.startswith(".")
            ])
            for exercise_dir in exercise_dirs:
                ex_path = os.path.join(subject_path, exercise_dir)
                # Only cam1.mp4 in multi-subject dataset
                v = os.path.join(ex_path, "cam1.mp4")
                if not os.path.exists(v):
                    continue
                gt = v.replace(".mp4", "_p2d.txt")
                entry = (v, gt, subject_dir, exercise_dir)
                if os.path.exists(gt):
                    pairs.append(entry)
                else:
                    skipped_no_gt.append(entry)

    return pairs, skipped_no_gt, fmt


# ==============================================================================
# Per-Video Plot Helper  (ไม่เปลี่ยน)
# ==============================================================================
def save_video_plots(csv_path, trajectory_path, error_path, j0_name, j1_name, j2_name):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"      [WARN] Cannot generate plots for {csv_path}: {e}")
        return

    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    # Angle trajectory
    plt.figure(figsize=(12, 4.5))
    plt.plot(df["frame"], df["gt_angle"], label="OptiTrack (GT)", color="#00DC50", lw=2.5, zorder=4)
    for col, label, color in [
        ("yolo_angle",      "YOLOv8-Pose",     "#FF0000"),
        ("movenet_angle",   "MoveNet (Thunder)","#FF8C00"),
        ("mediapipe_angle", "MediaPipe (Heavy)","#00C8C8"),
    ]:
        if col in df.columns:
            plt.plot(df["frame"], df[col], label=label, color=color, lw=1.5, alpha=0.85)
    plt.title(f"Joint Angle Trajectory ({j1_name} Angle)", fontsize=11, fontweight="bold")
    plt.xlabel("Frame"); plt.ylabel("Angle (°)")
    plt.grid(True, ls="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(trajectory_path, dpi=160); plt.close()

    # Spatial error bar chart
    joints = [j0_name.lower(), j1_name.lower(), j2_name.lower()]
    models = ["YOLOv8", "MoveNet", "MediaPipe"]
    err_data = {}
    for m in models:
        m_lower = "mediapipe" if m == "MediaPipe" else m.lower()
        err_data[m] = [
            df[f"{m_lower}_{j}_err_px"].dropna().mean() if f"{m_lower}_{j}_err_px" in df.columns else 0.0
            for j in joints
        ]

    x, w = np.arange(len(joints)), 0.25
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (m, color) in enumerate(zip(models, ["#FF5252", "#FF9800", "#00BCD4"])):
        bars = ax.bar(x + (i - 1) * w, err_data[m], w, label=m, color=color, edgecolor="black", lw=0.7)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.1f}px", xy=(bar.get_x() + w / 2, h),
                            xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Mean Spatial Error (px)"); ax.set_xticks(x)
    ax.set_xticklabels([j.capitalize() for j in joints])
    ax.set_title("Mean Spatial Coordinate Error by Joint & Model", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5, axis="y"); ax.legend()
    plt.tight_layout(); plt.savefig(error_path, dpi=160); plt.close()


# ==============================================================================
# Core Video Processor  (inference logic ไม่เปลี่ยน — เพิ่มแค่ subject_id + exercise_id param)
# ==============================================================================
def process_video_pair(
    video_path, gt_path,
    subject_id, exercise_id,           # ← passed directly, no path-parsing needed
    yolo_model, movenet_interp, mp_model_path,
    out_folder,
    limit_frames=None, save_video=False, save_plots=True,
    rotate_ids=None, device="mps",
):
    video_base = os.path.splitext(os.path.basename(video_path))[0]

    out_csv_path   = os.path.join(out_folder, f"{video_base}_report.csv")
    out_traj_plot  = os.path.join(out_folder, f"{video_base}_angle_trajectory.png")
    out_err_plot   = os.path.join(out_folder, f"{video_base}_spatial_error.png")
    if save_video:
        out_video_path = os.path.join(out_folder, f"{video_base}_annotated.mp4")

    # ── Ground Truth ──────────────────────────────────────────────────────────
    gt_coords = []
    with open(gt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) >= 6:
                gt_coords.append({
                    "j0": (parts[0], parts[1]),
                    "j1": (parts[2], parts[3]),
                    "j2": (parts[4], parts[5]),
                })
    if not gt_coords:
        return None

    # ── Video open ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_frames = min(total_frames, len(gt_coords))
    if limit_frames is not None:
        num_frames = min(num_frames, limit_frames)

    # ── Exercise metadata (from passed exercise_id — no path parsing) ─────────
    ex_name, ex_position, selected_side, region = get_exercise_info(exercise_id)
    is_upper = (region == "upper")

    if rotate_ids is None:
        rotate_ids = ROTATED_EXERCISE_IDS
    needs_rotation = exercise_id in rotate_ids
    rotation_type  = "90_cw" if needs_rotation else None

    j0_name, j1_name, j2_name = ("Shoulder", "Elbow", "Wrist") if is_upper else ("Hip", "Knee", "Ankle")

    if selected_side == "left":
        idx_s, idx_e, idx_w        = (5, 7, 9)  if is_upper else (11, 13, 15)
        mp_idx_s, mp_idx_e, mp_idx_w = (11, 13, 15) if is_upper else (23, 25, 27)
    else:
        idx_s, idx_e, idx_w        = (6, 8, 10) if is_upper else (12, 14, 16)
        mp_idx_s, mp_idx_e, mp_idx_w = (12, 14, 16) if is_upper else (24, 26, 28)

    # ── MediaPipe session ─────────────────────────────────────────────────────
    BaseOptions         = mp.tasks.BaseOptions
    PoseLandmarker      = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode   = mp.tasks.vision.RunningMode

    mp_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_model_path),
        running_mode=VisionRunningMode.VIDEO,
    ))

    # ── MoveNet (LiteRT) ──────────────────────────────────────────────────────
    mn_in_idx  = movenet_interp.get_input_details()[0]["index"]
    mn_out_idx = movenet_interp.get_output_details()[0]["index"]

    # ── Optional video writer ─────────────────────────────────────────────────
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    records = []

    # ── Frame loop ────────────────────────────────────────────────────────────
    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gt_data  = gt_coords[frame_idx]
        gt_j0, gt_j1, gt_j2 = gt_data["j0"], gt_data["j1"], gt_data["j2"]
        gt_angle = calculate_angle(gt_j0, gt_j1, gt_j2)

        if needs_rotation:
            inf_frame        = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            proc_w, proc_h   = height, width
        else:
            inf_frame        = frame
            proc_w, proc_h   = width, height

        # ── 1. YOLOv8 ────────────────────────────────────────────────────────
        yolo_results = yolo_model(inf_frame, verbose=False, device=device)
        yolo_s = yolo_e = yolo_w = (float("nan"), float("nan"))
        yolo_angle = float("nan")
        if yolo_results and yolo_results[0].keypoints is not None:
            xy_tensors = yolo_results[0].keypoints.xy
            if xy_tensors.shape[0] > 0:
                kps = xy_tensors[0].cpu().numpy()
                valid = lambda kp: not (kp[0] < 1.0 and kp[1] < 1.0)
                if valid(kps[idx_s]) and valid(kps[idx_e]) and valid(kps[idx_w]):
                    rs, re, rw = tuple(kps[idx_s]), tuple(kps[idx_e]), tuple(kps[idx_w])
                    if needs_rotation:
                        yolo_s = map_rotated_kps_to_orig(rs[0], rs[1], width, height, rotation_type)
                        yolo_e = map_rotated_kps_to_orig(re[0], re[1], width, height, rotation_type)
                        yolo_w = map_rotated_kps_to_orig(rw[0], rw[1], width, height, rotation_type)
                    else:
                        yolo_s, yolo_e, yolo_w = rs, re, rw
                    yolo_angle = calculate_angle(yolo_s, yolo_e, yolo_w)

        # ── 2. MoveNet ────────────────────────────────────────────────────────
        mn_rgb   = cv2.cvtColor(cv2.resize(inf_frame, (256, 256)), cv2.COLOR_BGR2RGB)
        mn_input = np.expand_dims(mn_rgb, axis=0).astype(np.uint8)
        movenet_interp.set_tensor(mn_in_idx, mn_input)
        movenet_interp.invoke()
        mn_kps = movenet_interp.get_tensor(mn_out_idx)[0, 0]

        movenet_s = movenet_e = movenet_w = (float("nan"), float("nan"))
        movenet_angle = float("nan")
        CONF = 0.20
        if mn_kps[idx_s, 2] >= CONF and mn_kps[idx_e, 2] >= CONF and mn_kps[idx_w, 2] >= CONF:
            rs = (mn_kps[idx_s, 1] * proc_w, mn_kps[idx_s, 0] * proc_h)
            re = (mn_kps[idx_e, 1] * proc_w, mn_kps[idx_e, 0] * proc_h)
            rw = (mn_kps[idx_w, 1] * proc_w, mn_kps[idx_w, 0] * proc_h)
            if needs_rotation:
                movenet_s = map_rotated_kps_to_orig(rs[0], rs[1], width, height, rotation_type)
                movenet_e = map_rotated_kps_to_orig(re[0], re[1], width, height, rotation_type)
                movenet_w = map_rotated_kps_to_orig(rw[0], rw[1], width, height, rotation_type)
            else:
                movenet_s, movenet_e, movenet_w = rs, re, rw
            movenet_angle = calculate_angle(movenet_s, movenet_e, movenet_w)

        # ── 3. MediaPipe ──────────────────────────────────────────────────────
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inf_frame)
        mp_results = mp_landmarker.detect_for_video(mp_image, int(frame_idx * 1000 / fps))

        mp_s = mp_e = mp_w = (float("nan"), float("nan"))
        mp_angle = float("nan")
        if mp_results.pose_landmarks:
            lm = mp_results.pose_landmarks[0]
            rs = (lm[mp_idx_s].x * proc_w, lm[mp_idx_s].y * proc_h)
            re = (lm[mp_idx_e].x * proc_w, lm[mp_idx_e].y * proc_h)
            rw = (lm[mp_idx_w].x * proc_w, lm[mp_idx_w].y * proc_h)
            if needs_rotation:
                mp_s = map_rotated_kps_to_orig(rs[0], rs[1], width, height, rotation_type)
                mp_e = map_rotated_kps_to_orig(re[0], re[1], width, height, rotation_type)
                mp_w = map_rotated_kps_to_orig(rw[0], rw[1], width, height, rotation_type)
            else:
                mp_s, mp_e, mp_w = rs, re, rw
            mp_angle = calculate_angle(mp_s, mp_e, mp_w)

        # ── Error computation ─────────────────────────────────────────────────
        def ae(a, b):
            return abs(a - b) if not (np.isnan(a) or np.isnan(b)) else float("nan")

        records.append({
            "frame":              frame_idx,
            "gt_angle":           gt_angle,
            "yolo_angle":         yolo_angle,
            "yolo_angle_err":     ae(yolo_angle, gt_angle),
            f"yolo_{j0_name.lower()}_err_px":     compute_spatial_error(yolo_s, gt_j0),
            f"yolo_{j1_name.lower()}_err_px":     compute_spatial_error(yolo_e, gt_j1),
            f"yolo_{j2_name.lower()}_err_px":     compute_spatial_error(yolo_w, gt_j2),
            "movenet_angle":      movenet_angle,
            "movenet_angle_err":  ae(movenet_angle, gt_angle),
            f"movenet_{j0_name.lower()}_err_px":  compute_spatial_error(movenet_s, gt_j0),
            f"movenet_{j1_name.lower()}_err_px":  compute_spatial_error(movenet_e, gt_j1),
            f"movenet_{j2_name.lower()}_err_px":  compute_spatial_error(movenet_w, gt_j2),
            "mediapipe_angle":    mp_angle,
            "mediapipe_angle_err":ae(mp_angle, gt_angle),
            f"mediapipe_{j0_name.lower()}_err_px":compute_spatial_error(mp_s, gt_j0),
            f"mediapipe_{j1_name.lower()}_err_px":compute_spatial_error(mp_e, gt_j1),
            f"mediapipe_{j2_name.lower()}_err_px":compute_spatial_error(mp_w, gt_j2),
        })

    cap.release()
    if writer:
        writer.release()
    mp_landmarker.close()

    # ── Save per-video CSV ────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    df.to_csv(out_csv_path, index=False)

    if save_plots:
        save_video_plots(out_csv_path, out_traj_plot, out_err_plot, j0_name, j1_name, j2_name)

    # ── Build summary row ─────────────────────────────────────────────────────
    summary = {
        "subject_id":    subject_id,
        "exercise":      f"{exercise_id:02d}",
        "exercise_name": ex_name,
        "position":      ex_position,
        "side":          selected_side,
        "region":        region,
        "rotated":       needs_rotation,
        "video":         os.path.basename(video_path),
        "num_frames":    num_frames,
    }
    for m_name, prefix in [("YOLOv8", "yolo"), ("MoveNet", "movenet"), ("MediaPipe", "mediapipe")]:
        errs = df[f"{prefix}_angle_err"].dropna()
        summary[f"{m_name}_valid_pct"] = 100.0 * len(errs) / num_frames if num_frames > 0 else 0.0
        summary[f"{m_name}_mae"]       = errs.mean() if len(errs) > 0 else float("nan")
        summary[f"{m_name}_j0_err"]    = df[f"{prefix}_{j0_name.lower()}_err_px"].dropna().mean()
        summary[f"{m_name}_j1_err"]    = df[f"{prefix}_{j1_name.lower()}_err_px"].dropna().mean()
        summary[f"{m_name}_j2_err"]    = df[f"{prefix}_{j2_name.lower()}_err_px"].dropna().mean()

    return summary


# ==============================================================================
# Post-processing: EDA + Summary Plots  (ไม่เปลี่ยน)
# ==============================================================================
def generate_all_post_summaries(summary_csv_path, output_dir):
    if not os.path.exists(summary_csv_path):
        print(f"[ERROR] Summary CSV not found: {summary_csv_path}")
        return

    df     = pd.read_csv(summary_csv_path)
    models = ["YOLOv8", "MoveNet", "MediaPipe"]

    # Model performance table
    rows = []
    for model in models:
        rows.append({
            "Model":               model,
            "Detection_Rate_Mean": df[f"{model}_valid_pct"].mean(),
            "Detection_Rate_Std":  df[f"{model}_valid_pct"].std(),
            "Detection_Rate_Min":  df[f"{model}_valid_pct"].min(),
            "Detection_Rate_Max":  df[f"{model}_valid_pct"].max(),
            "MAE_Mean":            df[f"{model}_mae"].mean(),
            "MAE_Std":             df[f"{model}_mae"].std(),
            "MAE_Min":             df[f"{model}_mae"].min(),
            "MAE_Max":             df[f"{model}_mae"].max(),
            "MAE_Median":          df[f"{model}_mae"].median(),
        })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(os.path.join(output_dir, "model_performance_summary.csv"), index=False)

    # Exercise pivot tables
    ep = []
    for ex in df["exercise_name"].unique():
        ex_d = df[df["exercise_name"] == ex]
        for m in models:
            ep.append({
                "Exercise":           ex,
                "Model":              m,
                "Avg_MAE":            ex_d[f"{m}_mae"].mean(),
                "Avg_Detection_Rate": ex_d[f"{m}_valid_pct"].mean(),
            })
    ep_df = pd.DataFrame(ep)
    mae_pivot       = ep_df.pivot(index="Exercise", columns="Model", values="Avg_MAE")
    detection_pivot = ep_df.pivot(index="Exercise", columns="Model", values="Avg_Detection_Rate")
    mae_pivot.to_csv(os.path.join(output_dir, "mae_by_exercise.csv"))
    detection_pivot.to_csv(os.path.join(output_dir, "detection_rate_by_exercise.csv"))

    # Global MAE bar chart
    mae_values = [df[f"{m}_mae"].dropna().mean() for m in models]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(models, mae_values, color=["#FF5252", "#FF9800", "#00BCD4"], edgecolor="black", width=0.45)
    plt.ylabel("Global MAE (Degrees)"); plt.title("Global Kinematic Angle Error (MAE)", fontsize=11, fontweight="bold")
    plt.grid(ls="--", alpha=0.5, axis="y")
    for bar in bars:
        h = bar.get_height()
        plt.annotate(f"{h:.2f}°", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="semibold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_model_comparison.png"), dpi=200); plt.close()

    # 9-panel EDA figure
    fig = plt.figure(figsize=(18, 18))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    w = 0.25

    plt.subplot(3, 3, 1)
    rates = [df[f"{m}_valid_pct"].mean() for m in models]
    bs = plt.bar(models, rates, color=colors, alpha=0.85, edgecolor="black")
    plt.title("Average Detection Rate (%)", fontweight="bold"); plt.ylim(0, 110)
    for b, r in zip(bs, rates):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()+1, f"{r:.1f}%", ha="center", va="bottom", fontweight="bold")

    plt.subplot(3, 3, 2)
    bs = plt.bar(models, mae_values, color=colors, alpha=0.85, edgecolor="black")
    plt.title("Average MAE (Degrees)", fontweight="bold")
    for b, m in zip(bs, mae_values):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()+0.5, f"{m:.2f}°", ha="center", va="bottom", fontweight="bold")

    plt.subplot(3, 3, 3)
    counts = df["exercise_name"].value_counts()
    plt.pie(counts.values, labels=counts.index, autopct="%1.0f%%", startangle=90, textprops={"fontsize": 7})
    plt.title("Exercise Distribution", fontweight="bold")

    plt.subplot(3, 3, 4)
    mae_ex = np.array([[df[df["exercise_name"] == ex][f"{m}_mae"].mean() for m in models]
                        for ex in sorted(df["exercise_name"].unique())])
    ex_labels = [ex[:18]+"…" if len(ex) > 20 else ex for ex in sorted(df["exercise_name"].unique())]
    x = np.arange(len(ex_labels))
    for i, (m, c) in enumerate(zip(models, colors)):
        plt.bar(x + (i-1)*w, mae_ex[:, i], w, label=m, color=c)
    plt.title("MAE by Exercise Type", fontweight="bold")
    plt.xticks(x, ex_labels, rotation=40, ha="right", fontsize=7); plt.legend(fontsize=8)

    plt.subplot(3, 3, 5)
    plt.boxplot([df[f"{m}_valid_pct"] for m in models], tick_labels=models,
                patch_artist=True, boxprops=dict(facecolor="#D0E8FF"))
    plt.title("Detection Rate Distribution"); plt.ylabel("Rate (%)"); plt.grid(axis="y", alpha=0.3)

    plt.subplot(3, 3, 6)
    plt.boxplot([df[f"{m}_mae"].dropna() for m in models], tick_labels=models,
                patch_artist=True, boxprops=dict(facecolor="#FFD0D0"))
    plt.title("MAE Distribution (Degrees)"); plt.ylabel("MAE"); plt.grid(axis="y", alpha=0.3)

    plt.subplot(3, 3, 7)
    reg = df.groupby("region")[[f"{m}_mae" for m in models]].mean()
    xr  = np.arange(len(reg))
    for i, (m, c) in enumerate(zip(models, colors)):
        plt.bar(xr+(i-1)*w, reg[f"{m}_mae"], w, label=m, color=c)
    plt.title("MAE by Body Region"); plt.xticks(xr, reg.index); plt.legend(fontsize=8)

    plt.subplot(3, 3, 8)
    pos = df.groupby("position")[[f"{m}_mae" for m in models]].mean()
    xp  = np.arange(len(pos))
    for i, (m, c) in enumerate(zip(models, colors)):
        plt.bar(xp+(i-1)*w, pos[f"{m}_mae"], w, label=m, color=c)
    plt.title("MAE by Exercise Position"); plt.xticks(xp, pos.index); plt.legend(fontsize=8)

    plt.subplot(3, 3, 9)
    # Subject-level performance (only meaningful in multi-subject dataset)
    if "subject_id" in df.columns and df["subject_id"].nunique() > 1:
        subj_mae = df.groupby("subject_id")[[f"{m}_mae" for m in models]].mean()
        xs = np.arange(len(subj_mae))
        for i, (m, c) in enumerate(zip(models, colors)):
            plt.bar(xs+(i-1)*w, subj_mae[f"{m}_mae"], w, label=m, color=c, alpha=0.8)
        plt.title("MAE by Subject", fontweight="bold")
        plt.xlabel("Subject ID"); plt.ylabel("Avg MAE (°)")
        plt.xticks(xs, subj_mae.index, fontsize=7, rotation=45)
        plt.legend(fontsize=8)
    else:
        side = df.groupby("side")[[f"{m}_mae" for m in models]].mean()
        xs   = np.arange(len(side))
        for i, (m, c) in enumerate(zip(models, colors)):
            plt.bar(xs+(i-1)*w, side[f"{m}_mae"], w, label=m, color=c)
        plt.title("MAE by Side (Left vs Right)"); plt.xticks(xs, side.index); plt.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_analysis_plots.png"), dpi=180); plt.close()

    # Terminal summary
    print("\n" + "="*80)
    print("                    FINAL MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(summary_df.round(2).to_string(index=False))
    print("\n" + "-"*80)
    print("MEAN ABSOLUTE ERROR BY EXERCISE (DEGREES):")
    print(mae_pivot.round(2).to_string())
    print("\n" + "-"*80)
    print(f"[SUCCESS] All artifacts saved → {os.path.abspath(output_dir)}")
    for f in ["batch_evaluation_summary.csv", "model_performance_summary.csv",
              "mae_by_exercise.csv", "detection_rate_by_exercise.csv",
              "global_model_comparison.png", "eda_analysis_plots.png"]:
        print(f"  • {f}")
    print("="*80 + "\n")


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Pose Evaluation — supports single-subject & multi-subject datasets"
    )
    parser.add_argument("--dir",            default="clips_mp4_cam1",
                        help="Dataset root.  Single-subject: clips_mp4/0  |  Multi-subject: clips_mp4_cam1")
    parser.add_argument("--yolo-model",     default="yolov8n-pose.pt",
                        help="YOLOv8-pose weights path or hub name")
    parser.add_argument("--movenet-model",  default="models/movenet_thunder.tflite",
                        help="Path to MoveNet TFLite model")
    parser.add_argument("--mediapipe-model",default="models/pose_landmarker_heavy.task",
                        help="Path to MediaPipe .task model")
    parser.add_argument("--output-dir",     default="output_comparison_results",
                        help="Directory for all output files")
    parser.add_argument("--limit-frames",   type=int, default=None,
                        help="Limit frames per video (for quick tests)")
    parser.add_argument("--save-videos",    action="store_true", default=False,
                        help="Save annotated output videos (slower)")
    parser.add_argument("--no-plots",       action="store_true", default=False,
                        help="Skip per-video plots (faster)")
    parser.add_argument("--rotate-ids",     type=str, default=None,
                        help="Comma-separated exercise IDs to rotate 90° (default: 3,4,7,8)")
    parser.add_argument("--device",         type=str, default=None, choices=["mps", "cpu", "cuda", "0"],
                        help="Override PyTorch device")
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        sys.exit(f"[ERROR] Dataset directory not found: {args.dir}")

    # ── Models ────────────────────────────────────────────────────────────────
    movenet_path, mp_model_path = ensure_model_files("models")
    if not os.path.exists(args.movenet_model):
        args.movenet_model = movenet_path
    if not os.path.exists(args.mediapipe_model):
        args.mediapipe_model = mp_model_path

    device = detect_device(force=args.device)

    if args.rotate_ids is None:
        rotate_ids = ROTATED_EXERCISE_IDS
    elif args.rotate_ids.strip() == "":
        rotate_ids = set()
    else:
        rotate_ids = {int(x.strip()) for x in args.rotate_ids.split(",") if x.strip()}

    # ── Discover pairs ────────────────────────────────────────────────────────
    print(f"\n[INFO] Scanning dataset: {args.dir}")
    pairs, skipped, fmt = discover_pairs(args.dir)
    print(f"[INFO] Detected format   : {fmt}")
    print(f"[INFO] Valid pairs found : {len(pairs)}")
    print(f"[INFO] Skipped (no GT)   : {len(skipped)}")
    if skipped:
        for v, gt, sid, ex in skipped[:5]:
            print(f"       ↳ subject={sid} exercise={ex} — {os.path.basename(v)}")
        if len(skipped) > 5:
            print(f"       ↳ ... and {len(skipped)-5} more")

    if not pairs:
        sys.exit("[ERROR] No valid video/GT pairs found. Check --dir path.")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  LOADING MODELS")
    print("="*80)
    print(f"[1/3] YOLOv8-pose ({args.yolo_model}) on {device.upper()}...")
    yolo_model = YOLO(args.yolo_model)

    print(f"[2/3] MoveNet Thunder ({args.movenet_model}) via LiteRT...")
    movenet_interp = Interpreter(model_path=args.movenet_model)
    movenet_interp.allocate_tensors()

    print(f"[3/3] MediaPipe Pose Landmarker ({args.mediapipe_model})")
    print("="*80 + "\n")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    results    = []
    start_time = time.time()
    save_plots = not args.no_plots

    progress = tqdm(pairs, total=len(pairs), desc="Evaluating", unit="video") if tqdm else pairs

    for idx, (v_path, gt_path, subject_id, exercise_dir) in enumerate(progress, start=1):

        # Parse exercise_id from the folder name (e.g. "01" → 1)
        try:
            exercise_id = int(exercise_dir)
        except ValueError:
            msg = f"[WARN] Cannot parse exercise_id from folder '{exercise_dir}', skipping."
            (tqdm.write if tqdm else print)(msg)
            continue

        # Output folder: output_dir/subject_<id>/exercise_dir
        if fmt == "multi-subject":
            out_folder = os.path.join(args.output_dir, f"subject_{subject_id.zfill(2)}", exercise_dir)
        else:
            out_folder = os.path.join(args.output_dir, exercise_dir)
        os.makedirs(out_folder, exist_ok=True)

        video_base    = os.path.splitext(os.path.basename(v_path))[0]
        expected_csv  = os.path.join(out_folder, f"{video_base}_report.csv")

        # ── Resume: skip if CSV already exists ────────────────────────────────
        if os.path.exists(expected_csv):
            msg = f"[{idx}/{len(pairs)}] SKIP (done): subject={subject_id} ex={exercise_dir}"
            (tqdm.write if tqdm else print)(msg)
            try:
                df_ex = pd.read_csv(expected_csv)
                ex_name, ex_pos, ex_side, ex_reg = get_exercise_info(exercise_id)
                is_u = (ex_reg == "upper")
                j0n, j1n, j2n = ("Shoulder","Elbow","Wrist") if is_u else ("Hip","Knee","Ankle")
                num_f = len(df_ex)
                skip_row = {
                    "subject_id":    subject_id,
                    "exercise":      f"{exercise_id:02d}",
                    "exercise_name": ex_name,
                    "position":      ex_pos,
                    "side":          ex_side,
                    "region":        ex_reg,
                    "rotated":       exercise_id in rotate_ids,
                    "video":         os.path.basename(v_path),
                    "num_frames":    num_f,
                }
                for m_name, prefix in [("YOLOv8","yolo"),("MoveNet","movenet"),("MediaPipe","mediapipe")]:
                    errs = df_ex[f"{prefix}_angle_err"].dropna() if f"{prefix}_angle_err" in df_ex.columns else pd.Series(dtype=float)
                    skip_row[f"{m_name}_valid_pct"] = 100.0 * len(errs) / num_f if num_f > 0 else 0.0
                    skip_row[f"{m_name}_mae"]       = errs.mean() if len(errs) > 0 else float("nan")
                    skip_row[f"{m_name}_j0_err"]    = df_ex[f"{prefix}_{j0n.lower()}_err_px"].dropna().mean()
                    skip_row[f"{m_name}_j1_err"]    = df_ex[f"{prefix}_{j1n.lower()}_err_px"].dropna().mean()
                    skip_row[f"{m_name}_j2_err"]    = df_ex[f"{prefix}_{j2n.lower()}_err_px"].dropna().mean()
                results.append(skip_row)
            except Exception:
                pass
            continue

        # ── Process ───────────────────────────────────────────────────────────
        t0  = time.time()
        res = process_video_pair(
            video_path=v_path, gt_path=gt_path,
            subject_id=subject_id, exercise_id=exercise_id,
            yolo_model=yolo_model, movenet_interp=movenet_interp,
            mp_model_path=args.mediapipe_model,
            out_folder=out_folder,
            limit_frames=args.limit_frames,
            save_video=args.save_videos,
            save_plots=save_plots,
            rotate_ids=rotate_ids,
            device=device,
        )

        if res:
            results.append(res)
            dt  = time.time() - t0
            msg = (f"  [sub={subject_id} ex={exercise_dir}] {res['exercise_name'][:30]} "
                   f"| MP: {res['MediaPipe_mae']:.2f}° YOLO: {res['YOLOv8_mae']:.2f}° "
                   f"MN: {res['MoveNet_mae']:.2f}° ({dt:.1f}s)")
            (tqdm.write if tqdm else print)(msg)

    if not results:
        print("[ERROR] No videos were processed.")
        return

    # ── Consolidated summary ──────────────────────────────────────────────────
    df_results      = pd.DataFrame(results)
    summary_csv     = os.path.join(args.output_dir, "batch_evaluation_summary.csv")
    df_results.to_csv(summary_csv, index=False)

    # ── Markdown report ───────────────────────────────────────────────────────
    md_rows = [
        f"| {r.get('subject_id','—')}/{r['exercise']}/{r['video']} "
        f"| {r['exercise_name']} | {r['side'].upper()} "
        f"| {r['YOLOv8_mae']:.2f}° | {r['MoveNet_mae']:.2f}° | {r['MediaPipe_mae']:.2f}° |"
        for _, r in df_results.iterrows()
    ]
    total_time = time.time() - start_time
    md = f"""# Batch Evaluation Report — Multi-Subject Dataset
* **Dataset format:** {fmt}
* **Videos evaluated:** {len(df_results)}
* **Total time:** {total_time/60:.1f} min

## Global Model Summary
| Model | Detection Rate | Global MAE |
|:---|:---:|:---:|
| **YOLOv8-pose** | {df_results['YOLOv8_valid_pct'].dropna().mean():.1f}% | {df_results['YOLOv8_mae'].dropna().mean():.2f}° |
| **MoveNet (Thunder)** | {df_results['MoveNet_valid_pct'].dropna().mean():.1f}% | {df_results['MoveNet_mae'].dropna().mean():.2f}° |
| **MediaPipe (Heavy)** | {df_results['MediaPipe_valid_pct'].dropna().mean():.1f}% | {df_results['MediaPipe_mae'].dropna().mean():.2f}° |

## Per-Video Results
| Subject/Exercise/Video | Exercise | Side | YOLOv8 MAE | MoveNet MAE | MediaPipe MAE |
|:---|:---|:---:|:---:|:---:|:---:|
{chr(10).join(md_rows)}
"""
    with open(os.path.join(args.output_dir, "batch_comparison_report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    generate_all_post_summaries(summary_csv, args.output_dir)


if __name__ == "__main__":
    main()
