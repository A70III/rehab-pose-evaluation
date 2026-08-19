import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

# Limit TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# MediaPipe expects legacy protobuf functions, patch if newer protobuf is installed.
try:
    import google.protobuf.message_factory as _message_factory
    import google.protobuf.symbol_database as _symbol_database

    if not hasattr(_message_factory.MessageFactory, "GetPrototype") and hasattr(_message_factory, "GetMessageClass"):
        _message_factory.MessageFactory.GetPrototype = (
            lambda self, descriptor: _message_factory.GetMessageClass(descriptor)
        )
    
    if not hasattr(_symbol_database.SymbolDatabase, "GetPrototype") and hasattr(_message_factory, "GetMessageClass"):
        _symbol_database.SymbolDatabase.GetPrototype = (
            lambda self, descriptor: _message_factory.GetMessageClass(descriptor)
        )
except Exception:
    pass

import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# Exercise Metadata (fixed, dataset-defined — see Table 1)
# ------------------------------------------------------------------------------
# ID -> (name, position, side, region)
EXERCISE_INFO = {
    1:  ("Bending the knee without support while sitting", "Seated",   "left",  "lower"),
    2:  ("Bending the knee with support while sitting",     "Seated",   "left",  "lower"),
    3:  ("Lift the extended leg",                            "Supine",   "left",  "lower"),
    4:  ("Bending the knee with bed support",                "Supine",   "left",  "lower"),
    5:  ("Bending the knee without support while sitting",  "Seated",   "right", "lower"),
    6:  ("Bending the knee with support while sitting",     "Seated",   "right", "lower"),
    7:  ("Lift the extended leg",                            "Supine",   "right", "lower"),
    8:  ("Bending the knee with bed support",                "Supine",   "right", "lower"),
    9:  ("Shoulder flexion",                                 "Seated",   "left",  "upper"),
    10: ("Horizontal weighted openings",                     "Standing", "left",  "upper"),
    11: ("External rotation of shoulders with elastic band", "Standing", "left",  "upper"),
    12: ("Circular pendulum",                                "Standing", "left",  "upper"),
    13: ("Shoulder flexion",                                 "Seated",   "right", "upper"),
    14: ("Horizontal weighted openings",                     "Standing", "right", "upper"),
    15: ("External rotation of shoulders with elastic band", "Standing", "right", "upper"),
    16: ("Circular pendulum",                                "Standing", "right", "upper"),
}

# Exercise IDs for which the source video is recorded/needs rotation.
# Toggle this set (or pass --no-rotate / --rotate-ids) to A/B test whether
# rotating the frame before running pose models changes accuracy.
ROTATED_EXERCISE_IDS = {3, 4, 7, 8}

# ------------------------------------------------------------------------------
# GPU Detection Helpers
# ------------------------------------------------------------------------------
def detect_yolo_device():
    """Return 0 if a CUDA GPU is available for PyTorch (used by YOLO), else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[GPU] PyTorch sees CUDA device: {name} -> YOLOv8 will run on GPU.")
            return 0
        print("[GPU] PyTorch reports no CUDA device available -> YOLOv8 will run on CPU.")
        return "cpu"
    except Exception as e:
        print(f"[GPU] Could not query PyTorch CUDA status ({e}) -> YOLOv8 will run on CPU.")
        return "cpu"

def report_tf_gpu_status():
    """Print whether TensorFlow (used by MoveNet) can see a GPU. TF picks it up automatically."""
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[GPU] TensorFlow sees {len(gpus)} GPU(s): {[g.name for g in gpus]} -> MoveNet will run on GPU.")
        else:
            print("[GPU] TensorFlow sees no GPU -> MoveNet will run on CPU. "
                  "(On native Windows, TF dropped GPU support after 2.10 — "
                  "use tensorflow==2.10.0, or run this script under WSL2 for a current TF+CUDA build.)")
    except Exception as e:
        print(f"[GPU] Could not query TensorFlow GPU status: {e}")


def calculate_angle(a, b, c):
    """Compute flexion angle (in degrees) formed by vector segments BA and BC."""
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(c, dtype=float)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return float("nan")
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))

def compute_spatial_error(pt_pred, pt_gt):
    """Euclidean distance in pixels between predicted and ground-truth coordinates."""
    if pt_pred is None or pt_gt is None:
        return float("nan")
    if any(np.isnan(pt_pred)) or any(np.isnan(pt_gt)):
        return float("nan")
    if (pt_pred[0] == 0.0 and pt_pred[1] == 0.0) or (pt_gt[0] == 0.0 and pt_gt[1] == 0.0):
        return float("nan")
    return float(np.linalg.norm(np.array(pt_pred[:2]) - np.array(pt_gt[:2])))

def map_rotated_kps_to_orig(x, y, orig_w, orig_h, rotation):
    """Map coordinates from a rotated image frame back to original image space."""
    if rotation == "90_cw":
        return y, orig_h - x
    elif rotation == "90_ccw":
        return orig_w - y, x
    return x, y

def get_exercise_info(exercise_id):
    """Look up fixed exercise metadata. Raises if the ID is not in the dataset's 16 cases."""
    if exercise_id not in EXERCISE_INFO:
        raise ValueError(
            f"Exercise ID {exercise_id} is not a recognized dataset exercise (expected 1-16)."
        )
    return EXERCISE_INFO[exercise_id]

# ------------------------------------------------------------------------------
# Visualization Helpers
# ------------------------------------------------------------------------------
def draw_pose(frame, j0, j1, j2, angle, color, label, text_offset_y):
    """Draw skeleton lines, joint circles, and angle text on the frame."""
    if any(np.isnan(v) for pt in [j0, j1, j2] for v in pt[:2]):
        return
        
    pt0 = tuple(map(int, j0[:2]))
    pt1 = tuple(map(int, j1[:2]))
    pt2 = tuple(map(int, j2[:2]))
    
    cv2.line(frame, pt0, pt1, color, 2, cv2.LINE_AA)
    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    
    for pt in (pt0, pt1, pt2):
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA)
        
    if not np.isnan(angle):
        text = f"{label}: {angle:.1f} deg"
        pos = (pt1[0] + 12, pt1[1] + text_offset_y)
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# ------------------------------------------------------------------------------
# Individual Video Plotting Helper
# ------------------------------------------------------------------------------
def save_video_plots(csv_path, trajectory_path, error_path, j0_name, j1_name, j2_name):
    """Generate and save comparison plots for a single video."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"      [WARN] Failed to generate plots for {csv_path}: {e}")
        return

    # Set up matplotlib style
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False

    # 1. Plot Angle Trajectories
    plt.figure(figsize=(12, 5))
    plt.plot(df["frame"], df["gt_angle"], label="OptiTrack (GT)", color="#00DC50", linewidth=2.5, zorder=4)
    if "yolo_angle" in df.columns:
        plt.plot(df["frame"], df["yolo_angle"], label="YOLOv8-Pose", color="#FF0000", linewidth=1.5, alpha=0.85)
    if "movenet_angle" in df.columns:
        plt.plot(df["frame"], df["movenet_angle"], label="MoveNet (Thunder)", color="#FF8C00", linewidth=1.5, alpha=0.85)
    if "mediapipe_angle" in df.columns:
        plt.plot(df["frame"], df["mediapipe_angle"], label="MediaPipe (Heavy)", color="#00C8C8", linewidth=1.5, alpha=0.85)
        
    plt.title(f"Joint Angle Trajectory Comparison ({j1_name} Flexion/Extension)", fontsize=12, fontweight="bold")
    plt.xlabel("Frame Number", fontsize=10)
    plt.ylabel("Angle (Degrees)", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")
    plt.tight_layout()
    plt.savefig(trajectory_path, dpi=200)
    plt.close()

    # 2. Plot Spatial Tracking Errors
    joints = [j0_name.lower(), j1_name.lower(), j2_name.lower()]
    models = ["YOLOv8", "MoveNet", "MediaPipe"]
    
    error_data = {}
    for m in models:
        error_data[m] = []
        m_lower = "mediapipe" if m == "MediaPipe" else m.lower()
        for j in joints:
            col_name = f"{m_lower}_{j}_err_px"
            if col_name in df.columns:
                mean_err = df[col_name].dropna().mean()
                error_data[m].append(mean_err if not np.isnan(mean_err) else 0.0)
            else:
                error_data[m].append(0.0)
                
    x = np.arange(len(joints))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width, error_data["YOLOv8"], width, label="YOLOv8-Pose", color="#FF5252", edgecolor="black", linewidth=0.7)
    rects2 = ax.bar(x, error_data["MoveNet"], width, label="MoveNet (Thunder)", color="#FF9800", edgecolor="black", linewidth=0.7)
    rects3 = ax.bar(x + width, error_data["MediaPipe"], width, label="MediaPipe (Heavy)", color="#00BCD4", edgecolor="black", linewidth=0.7)
    
    ax.set_ylabel("Mean Spatial Error (Pixels)", fontsize=10)
    ax.set_title("Mean Spatial Coordinate Error by Joint & Model", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([j.capitalize() for j in joints], fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax.legend(title="Pose Model", frameon=True, facecolor="white", edgecolor="gray")
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.1f} px",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
                            
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    max_err = max(max(error_data[m]) for m in models)
    if max_err > 0:
        plt.ylim(0, max_err * 1.15)
    plt.tight_layout()
    plt.savefig(error_path, dpi=200)
    plt.close()

# ------------------------------------------------------------------------------
# Core Processor for Single Video Pair
# ------------------------------------------------------------------------------
def process_video_pair(video_path, gt_path, yolo_model, movenet_fn, mp_model_path, out_folder,
                        limit_frames=None, save_video=True, save_plots=True, rotate_ids=None,
                        yolo_device=None):
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    
    # Paths for outputs
    out_video_path = os.path.join(out_folder, f"{video_base}_annotated.mp4")
    out_csv_path = os.path.join(out_folder, f"{video_base}_report.csv")
    out_md_path = os.path.join(out_folder, f"{video_base}_report.md")
    out_traj_plot = os.path.join(out_folder, f"{video_base}_angle_trajectory.png")
    out_err_plot = os.path.join(out_folder, f"{video_base}_spatial_error.png")

    # Load Ground Truth
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
                    "j2": (parts[4], parts[5])
                })
                
    if not gt_coords:
        return None

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    num_frames_to_process = min(total_frames, len(gt_coords))
    if limit_frames is not None:
        num_frames_to_process = min(num_frames_to_process, limit_frames)
        
    if yolo_device is None:
        yolo_device = "cpu"

    # --------------------------------------------------------------------
    # Exercise ID detection + fixed metadata lookup (no error-based guessing)
    # --------------------------------------------------------------------
    exercise_id = None
    norm_path = os.path.normpath(video_path).replace("\\", "/")
    parts = norm_path.split("/")
    for part in parts:
        if part.isdigit() and 1 <= int(part) <= 16:
            exercise_id = int(part)
            break

    if exercise_id is None:
        raise ValueError(
            f"Could not determine exercise ID from path: {video_path}. "
            f"Expected a folder named '01'-'16' somewhere in the path."
        )

    ex_name, ex_position, selected_side, region = get_exercise_info(exercise_id)
    is_upper = (region == "upper")

    # Rotation is decided by exercise ID membership in rotate_ids (defaults to
    # ROTATED_EXERCISE_IDS), not auto-detected.
    if rotate_ids is None:
        rotate_ids = ROTATED_EXERCISE_IDS
    needs_rotation = exercise_id in rotate_ids
    rotation_type = "90_cw" if needs_rotation else None

    if is_upper:
        j0_name, j1_name, j2_name = "Shoulder", "Elbow", "Wrist"
        angle_name = "Elbow Angle"
    else:
        j0_name, j1_name, j2_name = "Hip", "Knee", "Ankle"
        angle_name = "Knee Angle"

    body_part_label = f"Upper Body ({j0_name}-{j1_name}-{j2_name})" if is_upper else f"Lower Body ({j0_name}-{j1_name}-{j2_name})"

    # Set model joint indices directly from the fixed side (no detection pass needed)
    if selected_side == "left":
        idx_s, idx_e, idx_w = (5, 7, 9) if is_upper else (11, 13, 15)
        mp_idx_s, mp_idx_e, mp_idx_w = (11, 13, 15) if is_upper else (23, 25, 27)
    else:
        idx_s, idx_e, idx_w = (6, 8, 10) if is_upper else (12, 14, 16)
        mp_idx_s, mp_idx_e, mp_idx_w = (12, 14, 16) if is_upper else (24, 26, 28)
        
    # Setup MediaPipe Pose Landmarker for this video (fresh session, fresh timestamps)
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    mp_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_model_path),
        running_mode=VisionRunningMode.VIDEO
    )
    mp_landmarker = PoseLandmarker.create_from_options(mp_options)
        
    # Setup Video Writer if saving
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
        
    # Colors
    COLOR_GT = (0, 220, 80)
    COLOR_YOLO = (0, 0, 255)
    COLOR_MOVENET = (255, 140, 0)
    COLOR_MP = (0, 240, 240)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    records = []

    frame_iter = range(num_frames_to_process)
    if tqdm is not None:
        frame_iter = tqdm(
            frame_iter,
            total=num_frames_to_process,
            desc=f"Frames {video_base}",
            unit="frame",
            leave=False,
            dynamic_ncols=True,
        )

    for frame_idx in frame_iter:
        ret, frame = cap.read()
        if not ret:
            break
            
        gt_data = gt_coords[frame_idx]
        gt_j0, gt_j1, gt_j2 = gt_data["j0"], gt_data["j1"], gt_data["j2"]
        gt_angle = calculate_angle(gt_j0, gt_j1, gt_j2)
        
        if needs_rotation:
            inf_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            proc_w, proc_h = height, width
        else:
            inf_frame = frame
            proc_w, proc_h = width, height
            
        # YOLOv8
        yolo_results = yolo_model(inf_frame, verbose=False, device=yolo_device)
        yolo_s = yolo_e = yolo_w = (float("nan"), float("nan"))
        yolo_angle = float("nan")
        if yolo_results and yolo_results[0].keypoints is not None:
            xy_tensors = yolo_results[0].keypoints.xy
            if xy_tensors.shape[0] > 0:
                person_kps = xy_tensors[0].cpu().numpy()
                def is_valid_yolo(kp):
                    return not (kp[0] < 1.0 and kp[1] < 1.0)
                if is_valid_yolo(person_kps[idx_s]) and is_valid_yolo(person_kps[idx_e]) and is_valid_yolo(person_kps[idx_w]):
                    yolo_s_raw = tuple(person_kps[idx_s])
                    yolo_e_raw = tuple(person_kps[idx_e])
                    yolo_w_raw = tuple(person_kps[idx_w])
                    if needs_rotation:
                        yolo_s = map_rotated_kps_to_orig(yolo_s_raw[0], yolo_s_raw[1], width, height, rotation_type)
                        yolo_e = map_rotated_kps_to_orig(yolo_e_raw[0], yolo_e_raw[1], width, height, rotation_type)
                        yolo_w = map_rotated_kps_to_orig(yolo_w_raw[0], yolo_w_raw[1], width, height, rotation_type)
                    else:
                        yolo_s, yolo_e, yolo_w = yolo_s_raw, yolo_e_raw, yolo_w_raw
                    yolo_angle = calculate_angle(yolo_s, yolo_e, yolo_w)
                    
        # MoveNet
        movenet_resized = cv2.resize(inf_frame, (256, 256))
        movenet_rgb = cv2.cvtColor(movenet_resized, cv2.COLOR_BGR2RGB)
        movenet_input = tf.convert_to_tensor(np.expand_dims(movenet_rgb, axis=0), dtype=tf.int32)
        movenet_outputs = movenet_fn(movenet_input)
        movenet_kps = movenet_outputs['output_0'].numpy()[0, 0]
        
        movenet_s = movenet_e = movenet_w = (float("nan"), float("nan"))
        movenet_angle = float("nan")
        CONF_THRESH = 0.20
        if (movenet_kps[idx_s, 2] >= CONF_THRESH and 
            movenet_kps[idx_e, 2] >= CONF_THRESH and 
            movenet_kps[idx_w, 2] >= CONF_THRESH):
            movenet_s_raw = (movenet_kps[idx_s, 1] * proc_w, movenet_kps[idx_s, 0] * proc_h)
            movenet_e_raw = (movenet_kps[idx_e, 1] * proc_w, movenet_kps[idx_e, 0] * proc_h)
            movenet_w_raw = (movenet_kps[idx_w, 1] * proc_w, movenet_kps[idx_w, 0] * proc_h)
            if needs_rotation:
                movenet_s = map_rotated_kps_to_orig(movenet_s_raw[0], movenet_s_raw[1], width, height, rotation_type)
                movenet_e = map_rotated_kps_to_orig(movenet_e_raw[0], movenet_e_raw[1], width, height, rotation_type)
                movenet_w = map_rotated_kps_to_orig(movenet_w_raw[0], movenet_w_raw[1], width, height, rotation_type)
            else:
                movenet_s, movenet_e, movenet_w = movenet_s_raw, movenet_e_raw, movenet_w_raw
            movenet_angle = calculate_angle(movenet_s, movenet_e, movenet_w)
            
        # MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inf_frame)
        frame_timestamp_ms = int((frame_idx * 1000) / fps)
        mp_results = mp_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        mp_s = mp_e = mp_w = (float("nan"), float("nan"))
        mp_angle = float("nan")
        if mp_results.pose_landmarks:
            landmarks = mp_results.pose_landmarks[0]
            mp_s_raw = (landmarks[mp_idx_s].x * proc_w, landmarks[mp_idx_s].y * proc_h)
            mp_e_raw = (landmarks[mp_idx_e].x * proc_w, landmarks[mp_idx_e].y * proc_h)
            mp_w_raw = (landmarks[mp_idx_w].x * proc_w, landmarks[mp_idx_w].y * proc_h)
            if needs_rotation:
                mp_s = map_rotated_kps_to_orig(mp_s_raw[0], mp_s_raw[1], width, height, rotation_type)
                mp_e = map_rotated_kps_to_orig(mp_e_raw[0], mp_e_raw[1], width, height, rotation_type)
                mp_w = map_rotated_kps_to_orig(mp_w_raw[0], mp_w_raw[1], width, height, rotation_type)
            else:
                mp_s, mp_e, mp_w = mp_s_raw, mp_e_raw, mp_w_raw
            if (landmarks[mp_idx_s].visibility > 0.1 and 
                landmarks[mp_idx_e].visibility > 0.1 and 
                landmarks[mp_idx_w].visibility > 0.1):
                mp_angle = calculate_angle(mp_s, mp_e, mp_w)
            else:
                mp_s = mp_e = mp_w = (float("nan"), float("nan"))
                
        # Calculate errors
        yolo_angle_err = abs(yolo_angle - gt_angle) if not (np.isnan(yolo_angle) or np.isnan(gt_angle)) else float("nan")
        movenet_angle_err = abs(movenet_angle - gt_angle) if not (np.isnan(movenet_angle) or np.isnan(gt_angle)) else float("nan")
        mp_angle_err = abs(mp_angle - gt_angle) if not (np.isnan(mp_angle) or np.isnan(gt_angle)) else float("nan")
        
        yolo_j0_err = compute_spatial_error(yolo_s, gt_j0)
        yolo_j1_err = compute_spatial_error(yolo_e, gt_j1)
        yolo_j2_err = compute_spatial_error(yolo_w, gt_j2)
        
        movenet_j0_err = compute_spatial_error(movenet_s, gt_j0)
        movenet_j1_err = compute_spatial_error(movenet_e, gt_j1)
        movenet_j2_err = compute_spatial_error(movenet_w, gt_j2)
        
        mp_j0_err = compute_spatial_error(mp_s, gt_j0)
        mp_j1_err = compute_spatial_error(mp_e, gt_j1)
        mp_j2_err = compute_spatial_error(mp_w, gt_j2)
        
        records.append({
            "frame": frame_idx + 1,
            "gt_angle": round(gt_angle, 3) if not np.isnan(gt_angle) else None,
            "yolo_angle": round(yolo_angle, 3) if not np.isnan(yolo_angle) else None,
            "movenet_angle": round(movenet_angle, 3) if not np.isnan(movenet_angle) else None,
            "mediapipe_angle": round(mp_angle, 3) if not np.isnan(mp_angle) else None,
            
            "yolo_angle_err": round(yolo_angle_err, 3) if not np.isnan(yolo_angle_err) else None,
            "movenet_angle_err": round(movenet_angle_err, 3) if not np.isnan(movenet_angle_err) else None,
            "mediapipe_angle_err": round(mp_angle_err, 3) if not np.isnan(mp_angle_err) else None,
            
            f"yolo_{j0_name.lower()}_err_px": round(yolo_j0_err, 3) if not np.isnan(yolo_j0_err) else None,
            f"yolo_{j1_name.lower()}_err_px": round(yolo_j1_err, 3) if not np.isnan(yolo_j1_err) else None,
            f"yolo_{j2_name.lower()}_err_px": round(yolo_j2_err, 3) if not np.isnan(yolo_j2_err) else None,
            
            f"movenet_{j0_name.lower()}_err_px": round(movenet_j0_err, 3) if not np.isnan(movenet_j0_err) else None,
            f"movenet_{j1_name.lower()}_err_px": round(movenet_j1_err, 3) if not np.isnan(movenet_j1_err) else None,
            f"movenet_{j2_name.lower()}_err_px": round(movenet_j2_err, 3) if not np.isnan(movenet_j2_err) else None,
            
            f"mediapipe_{j0_name.lower()}_err_px": round(mp_j0_err, 3) if not np.isnan(mp_j0_err) else None,
            f"mediapipe_{j1_name.lower()}_err_px": round(mp_j1_err, 3) if not np.isnan(mp_j1_err) else None,
            f"mediapipe_{j2_name.lower()}_err_px": round(mp_j2_err, 3) if not np.isnan(mp_j2_err) else None,
        })
        
        if writer and save_video:
            # Overlays
            draw_pose(frame, gt_j0, gt_j1, gt_j2, gt_angle, COLOR_GT, "GT", text_offset_y=-30)
            draw_pose(frame, yolo_s, yolo_e, yolo_w, yolo_angle, COLOR_YOLO, "YOLO", text_offset_y=-10)
            draw_pose(frame, movenet_s, movenet_e, movenet_w, movenet_angle, COLOR_MOVENET, "MN", text_offset_y=10)
            draw_pose(frame, mp_s, mp_e, mp_w, mp_angle, COLOR_MP, "MP", text_offset_y=30)
            
            # HUD
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (330, 150), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            def draw_hud_text(text, y_pos, color=(255,255,255), scale=0.45):
                cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
                
            draw_hud_text(f"Frame: {frame_idx + 1} / {num_frames_to_process} ({selected_side.upper()} side)", 30, scale=0.5)
            draw_hud_text(f"GT {angle_name}  : {gt_angle:.1f} deg" if not np.isnan(gt_angle) else f"GT {angle_name}  : N/A", 55, COLOR_GT)
            draw_hud_text(f"YOLO Angle (Err): {yolo_angle:.1f} deg ({yolo_angle_err:.1f})" if not np.isnan(yolo_angle) else "YOLO Angle : N/A", 75, COLOR_YOLO)
            draw_hud_text(f"MoveNet (Err)   : {movenet_angle:.1f} deg ({movenet_angle_err:.1f})" if not np.isnan(movenet_angle) else "MoveNet : N/A", 95, COLOR_MOVENET)
            draw_hud_text(f"MediaPipe (Err) : {mp_angle:.1f} deg ({mp_angle_err:.1f})" if not np.isnan(mp_angle) else "MediaPipe : N/A", 115, COLOR_MP)
            
            # Legend
            cv2.rectangle(frame, (width - 180, height - 100), (width - 10, height - 10), (30,30,30), -1)
            def draw_legend(text, y_pos, color):
                cv2.circle(frame, (width - 165, y_pos - 5), 4, color, -1)
                cv2.putText(frame, text, (width - 150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                
            draw_legend("OptiTrack (GT)", height - 80, COLOR_GT)
            draw_legend("YOLOv8-Pose", height - 60, COLOR_YOLO)
            draw_legend("MoveNet (Thunder)", height - 40, COLOR_MOVENET)
            draw_legend("MediaPipe (Heavy)", height - 20, COLOR_MP)
            
            writer.write(frame)
            
    cap.release()
    if writer:
        writer.release()
        
    # Save CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv_path, index=False)
    
    # Save Plots
    if save_plots:
        save_video_plots(out_csv_path, out_traj_plot, out_err_plot, j0_name, j1_name, j2_name)
        
    # Compute Summary statistics
    summary = {
        "video": os.path.basename(video_path),
        "exercise": f"{exercise_id:02d}",
        "exercise_name": ex_name,
        "position": ex_position,
        "side": selected_side,
        "region": region,
        "rotated": needs_rotation,
    }
    
    y_angle_errs = df_out["yolo_angle_err"].dropna()
    mn_angle_errs = df_out["movenet_angle_err"].dropna()
    mp_angle_errs = df_out["mediapipe_angle_err"].dropna()
    
    for model_name, prefix, errs in [("YOLOv8", "yolo", y_angle_errs), 
                                     ("MoveNet", "movenet", mn_angle_errs), 
                                     ("MediaPipe", "mediapipe", mp_angle_errs)]:
        summary[f"{model_name}_valid_pct"] = 100.0 * len(errs) / num_frames_to_process if num_frames_to_process > 0 else 0.0
        summary[f"{model_name}_mae"] = errs.mean()
        summary[f"{model_name}_j0_err"] = df_out[f"{prefix}_{j0_name.lower()}_err_px"].dropna().mean()
        summary[f"{model_name}_j1_err"] = df_out[f"{prefix}_{j1_name.lower()}_err_px"].dropna().mean()
        summary[f"{model_name}_j2_err"] = df_out[f"{prefix}_{j2_name.lower()}_err_px"].dropna().mean()

    md_content = f"""# รายงานการเปรียบเทียบ {video_base}
    
- **ท่าออกกำลังกาย:** {exercise_id:02d} - {ex_name}
- **ตำแหน่งร่างกาย:** {body_part_label}
- **ลักษณะท่า (Position):** {ex_position}
- **ข้างที่วิเคราะห์ (กำหนดตามชุดข้อมูล):** {selected_side.upper()}
- **หมุนเฟรม 90° ก่อนประมวลผล:** {"ใช่" if needs_rotation else "ไม่"}
- **จำนวนเฟรมที่เปรียบเทียบ:** {num_frames_to_process} เฟรม

## สรุปค่าความคลาดเคลื่อนเฉลี่ย (Mean Errors)

| โมเดล | ความสำเร็จในการตรวจจับ | ความคลาดเคลื่อนมุมเฉลี่ย (MAE) | {j0_name} Error (px) | {j1_name} Error (px) | {j2_name} Error (px) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-pose** | {summary['YOLOv8_valid_pct']:.1f}% | {summary['YOLOv8_mae']:.2f}° | {summary['YOLOv8_j0_err']:.2f} px | {summary['YOLOv8_j1_err']:.2f} px | {summary['YOLOv8_j2_err']:.2f} px |
| **MoveNet (Thunder)** | {summary['MoveNet_valid_pct']:.1f}% | {summary['MoveNet_mae']:.2f}° | {summary['MoveNet_j0_err']:.2f} px | {summary['MoveNet_j1_err']:.2f} px | {summary['MoveNet_j2_err']:.2f} px |
| **MediaPipe (Heavy)** | {summary['MediaPipe_valid_pct']:.1f}% | {summary['MediaPipe_mae']:.2f}° | {summary['MediaPipe_j0_err']:.2f} px | {summary['MediaPipe_j1_err']:.2f} px | {summary['MediaPipe_j2_err']:.2f} px |
"""
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Close MediaPipe landmarker session
    mp_landmarker.close()
        
    return summary

# ------------------------------------------------------------------------------
# Global Summary Plotting Helper
# ------------------------------------------------------------------------------
def save_global_summary_plots(df, output_dir):
    """Generate global comparison plots across all videos."""
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # Global MAE Comparison per model
    models = ["YOLOv8", "MoveNet", "MediaPipe"]
    mae_values = [
        df["YOLOv8_mae"].dropna().mean(),
        df["MoveNet_mae"].dropna().mean(),
        df["MediaPipe_mae"].dropna().mean()
    ]
    
    plt.figure(figsize=(8, 5))
    colors = ["#FF5252", "#FF9800", "#00BCD4"]
    bars = plt.bar(models, mae_values, color=colors, edgecolor="black", width=0.5)
    
    plt.ylabel("Global Mean Absolute Error (Degrees)", fontsize=10)
    plt.title("Global Kinematic Angle Error (MAE) Comparison", fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f"{height:.2f}°",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='semibold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_model_comparison.png"), dpi=250)
    plt.close()

# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Structured Multi-Model Batch Evaluation Pipeline.")
    parser.add_argument("--dir", default=r"clips_mp4/0", help="Root directory containing exercise subfolders (01, 02, etc.)")
    parser.add_argument("--yolo-model", default=r"c:\Homework\LAB\YOLO\yolov8n-pose.pt", help="Path to YOLOv8-pose model weights")
    parser.add_argument("--movenet-model", default=r"C:\Users\LOQ\AppData\Local\Temp\tfhub_modules\f729a5f3231391676ca61cc7ab789993549d8bca", help="Path to MoveNet cached TFHub module")
    parser.add_argument("--mediapipe-model", default=r"c:\Homework\LAB\Medie\pose_landmarker_heavy.task", help="Path to MediaPipe Task file")
    parser.add_argument("--output-dir", default=r"output_comparison_results", help="Directory to save structured evaluation results")
    parser.add_argument("--limit-frames", type=int, default=None, help="Limit number of frames to process per video (e.g. 50, 100 for fast testing)")
    parser.add_argument("--save-videos", action="store_true", default=True, help="Save annotated videos (default)")
    parser.add_argument("--no-videos", action="store_false", dest="save_videos", help="Disable saving annotated videos to speed up processing")
    parser.add_argument("--save-plots", action="store_true", default=True, help="Save plots for each video (default)")
    parser.add_argument("--no-plots", action="store_false", dest="save_plots", help="Disable plotting for each video to speed up")
    parser.add_argument("--rotate-ids", type=str, default=None,
                         help="Comma-separated exercise IDs to rotate 90° before pose inference "
                              "(default: 3,4,7,8 — the Supine exercises). Pass an empty string '' "
                              "to disable rotation entirely for an A/B comparison.")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "gpu"],
                         help="Force YOLOv8 to run on 'cpu' or 'gpu', overriding auto-detection.")
    argv0 = os.path.basename(sys.argv[0])
    if argv0 in {"ipykernel_launcher.py", "colab_kernel_launcher.py"}:
        args, unknown_args = parser.parse_known_args()
        if unknown_args:
            print(f"[INFO] Ignoring notebook/kernel arguments: {unknown_args}")
    else:
        args = parser.parse_args()

    if not os.path.exists(args.dir):
        sys.exit(f"[ERROR] Source directory not found: {args.dir}")

    # Parse rotate-ids override
    if args.rotate_ids is None:
        rotate_ids = ROTATED_EXERCISE_IDS
    elif args.rotate_ids.strip() == "":
        rotate_ids = set()
    else:
        rotate_ids = {int(x.strip()) for x in args.rotate_ids.split(",") if x.strip()}
    print(f"[INFO] Rotation will be applied to exercise IDs: {sorted(rotate_ids) if rotate_ids else 'none'}")

    # GPU detection (once, before loading models)
    print("\n" + "="*75)
    print("  GPU AVAILABILITY CHECK")
    print("="*75)
    yolo_device = detect_yolo_device()
    if args.device == "cpu":
        yolo_device = "cpu"
        print("[GPU] Override: forcing YOLOv8 to CPU via --device cpu")
    elif args.device == "gpu":
        yolo_device = 0
        print("[GPU] Override: forcing YOLOv8 to GPU (device 0) via --device gpu")
    print("="*75 + "\n")

    # Discover all video/ground-truth pairs
    print(f"\n[INFO] Scanning for video/ground-truth pairs under: {args.dir}...")
    pairs = []
    
    # Get all subdirectories (like 01, 02, ..., 15)
    subdirs = sorted([d for d in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, d))])
    
    # Fallback to current folder if no subdirs
    if not subdirs:
        videos = glob.glob(os.path.join(args.dir, "cam*.mp4"))
        for v in videos:
            base, _ = os.path.splitext(v)
            gt = f"{base}_p2d.txt"
            if os.path.exists(gt):
                pairs.append((v, gt))
    else:
        for subdir in subdirs:
            subdir_path = os.path.join(args.dir, subdir)
            videos = glob.glob(os.path.join(subdir_path, "cam*.mp4"))
            for v in videos:
                base, _ = os.path.splitext(v)
                gt = f"{base}_p2d.txt"
                if os.path.exists(gt):
                    pairs.append((v, gt))
                    
    total_videos = len(pairs)
    print(f"[SUCCESS] Discovered {total_videos} video/ground-truth evaluation pairs.")
    if total_videos == 0:
        return

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Models Once
    print("\n" + "="*75)
    print("  LOADING DEEP LEARNING MODEL SUITE INTO RAM")
    print("="*75)
    
    print("[INFO] Loading YOLOv8-Pose...")
    yolo_model = YOLO(args.yolo_model)
    
    print("[INFO] Loading MoveNet SinglePose Thunder...")
    if os.path.exists(args.movenet_model):
        try:
            movenet = tf.saved_model.load(args.movenet_model)
            movenet_fn = movenet.signatures['serving_default']
        except Exception as e:
            print(f"[WARN] Failed to load MoveNet from local path: {e}. Attempting TF Hub load...")
            import tensorflow_hub as hub
            movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
            movenet_fn = movenet.signatures['serving_default']
    else:
        print("[INFO] Local MoveNet path not found. Attempting TF Hub load...")
        import tensorflow_hub as hub
        movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        movenet_fn = movenet.signatures['serving_default']

    report_tf_gpu_status()
    print("="*75 + "\n")

    # Main Processing Loop
    results = []
    start_time = time.time()
    
    print(f"[INFO] Beginning batch processing of {total_videos} videos...")
    progress_iter = pairs
    if tqdm is not None:
        progress_iter = tqdm(
            pairs,
            total=total_videos,
            desc="Processing videos",
            unit="video",
            dynamic_ncols=True,
        )

    def progress_log(message):
        if tqdm is not None:
            tqdm.write(message)
        else:
            print(message)

    for idx, (v_path, gt_path) in enumerate(progress_iter, start=1):
        # Extract folder name (e.g. "01") and video name (e.g. "cam0")
        parent_folder = os.path.basename(os.path.dirname(v_path))
        video_name = os.path.basename(v_path)
        if tqdm is not None:
            progress_iter.set_postfix_str(f"{parent_folder}/{video_name}", refresh=True)
        
        # Create folder-specific output directory
        target_out_folder = os.path.join(args.output_dir, parent_folder)
        os.makedirs(target_out_folder, exist_ok=True)

        # --- Resume: skip video if output CSV already exists ---
        video_base_name = os.path.splitext(video_name)[0]
        expected_csv = os.path.join(target_out_folder, f"{video_base_name}_report.csv")
        if os.path.exists(expected_csv):
            progress_log(f"[{idx}/{total_videos}] SKIP (already done): {parent_folder}/{video_name} — rebuilding summary row from existing CSV")
            try:
                df_existing = pd.read_csv(expected_csv)
                # Determine exercise_id from path
                norm_path = os.path.normpath(v_path).replace("\\", "/")
                _parts = norm_path.split("/")
                ex_id = None
                for _p in _parts:
                    if _p.isdigit() and 1 <= int(_p) <= 16:
                        ex_id = int(_p)
                        break
                if ex_id is None:
                    progress_log(f"      [WARN] Cannot determine exercise ID for {parent_folder}/{video_name}, skipping summary rebuild.")
                    continue
                ex_name, ex_position, ex_side, ex_region = get_exercise_info(ex_id)
                needs_rot = ex_id in rotate_ids
                is_upper = (ex_region == "upper")
                j0_name, j1_name, j2_name = ("Shoulder", "Elbow", "Wrist") if is_upper else ("Hip", "Knee", "Ankle")
                num_frames = len(df_existing)
                res_skip = {
                    "video": video_name,
                    "exercise": f"{ex_id:02d}",
                    "exercise_name": ex_name,
                    "position": ex_position,
                    "side": ex_side,
                    "region": ex_region,
                    "rotated": needs_rot,
                    "folder": parent_folder,
                }
                for model_name, prefix in [("YOLOv8", "yolo"), ("MoveNet", "movenet"), ("MediaPipe", "mediapipe")]:
                    angle_err_col = f"{prefix}_angle_err"
                    j0_col = f"{prefix}_{j0_name.lower()}_err_px"
                    j1_col = f"{prefix}_{j1_name.lower()}_err_px"
                    j2_col = f"{prefix}_{j2_name.lower()}_err_px"
                    errs = df_existing[angle_err_col].dropna() if angle_err_col in df_existing.columns else pd.Series(dtype=float)
                    res_skip[f"{model_name}_valid_pct"] = 100.0 * len(errs) / num_frames if num_frames > 0 else 0.0
                    res_skip[f"{model_name}_mae"] = errs.mean() if len(errs) > 0 else float("nan")
                    res_skip[f"{model_name}_j0_err"] = df_existing[j0_col].dropna().mean() if j0_col in df_existing.columns else float("nan")
                    res_skip[f"{model_name}_j1_err"] = df_existing[j1_col].dropna().mean() if j1_col in df_existing.columns else float("nan")
                    res_skip[f"{model_name}_j2_err"] = df_existing[j2_col].dropna().mean() if j2_col in df_existing.columns else float("nan")
                results.append(res_skip)
            except Exception as _e:
                progress_log(f"      [WARN] Could not rebuild summary for {parent_folder}/{video_name}: {_e}")
            continue

        progress_log(f"[{idx}/{total_videos}] Processing: {parent_folder}/{video_name} ...")
        t0 = time.time()
        
        try:
            res = process_video_pair(
                video_path=v_path,
                gt_path=gt_path,
                yolo_model=yolo_model,
                movenet_fn=movenet_fn,
                mp_model_path=args.mediapipe_model,
                out_folder=target_out_folder,
                limit_frames=args.limit_frames,
                save_video=args.save_videos,
                save_plots=args.save_plots,
                rotate_ids=rotate_ids,
                yolo_device=yolo_device
            )
        except ValueError as e:
            progress_log(f"      [WARN] Skipping {parent_folder}/{video_name}: {e}")
            res = None
        
        if res:
            res["folder"] = parent_folder
            results.append(res)
            dt = time.time() - t0
            print(f"      Success! [{res['exercise']}] {res['exercise_name']} | Side: {res['side'].upper()} | MP MAE: {res['MediaPipe_mae']:.2f}° | YOLO MAE: {res['YOLOv8_mae']:.2f}° | MN MAE: {res['MoveNet_mae']:.2f}° | Time: {dt:.1f}s")
        else:
            progress_log(f"      [WARN] Failed to process {parent_folder}/{video_name}")



    if not results:
        print("[ERROR] No videos were successfully processed.")
        return

    # Save consolidated report CSV
    df = pd.DataFrame(results)
    output_csv_path = os.path.join(args.output_dir, "batch_evaluation_summary.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"\n[SUCCESS] Saved consolidated batch report -> {output_csv_path}")

    # Generate global summary plots
    save_global_summary_plots(df, args.output_dir)
    print(f"[SUCCESS] Saved global comparison plots in output folder.")

    # Calculate global averages
    yolo_global_mae = df["YOLOv8_mae"].dropna().mean()
    movenet_global_mae = df["MoveNet_mae"].dropna().mean()
    mediapipe_global_mae = df["MediaPipe_mae"].dropna().mean()
    
    yolo_global_success = df["YOLOv8_valid_pct"].dropna().mean()
    movenet_global_success = df["MoveNet_valid_pct"].dropna().mean()
    mediapipe_global_success = df["MediaPipe_valid_pct"].dropna().mean()

    # Save a detailed Markdown report
    md_report_path = os.path.join(args.output_dir, "batch_comparison_report.md")
    
    md_rows = []
    for _, r in df.iterrows():
        md_rows.append(
            f"| {r['folder']}/{r['video']} | {r['exercise']} - {r['exercise_name']} | {r['side'].upper()} | {r['YOLOv8_mae']:.2f}° | {r['MoveNet_mae']:.2f}° | {r['MediaPipe_mae']:.2f}° |"
        )
        
    joined_rows = "\n".join(md_rows)
    md_content = f"""# รายงานการเปรียบเทียบประสิทธิภาพแบบกลุ่ม (Batch Multi-Model Comparison Report)

รายงานการประเมินความคลาดเคลื่อนของการวัดมุมข้อต่อและการติดตามพิกัดข้อต่อของโมเดล **YOLOv8-pose**, **MoveNet**, และ **MediaPipe** เทียบกับข้อมูลอ้างอิงจริง **OptiTrack Ground Truth** ครอบคลุมกล้องและท่ากายภาพบำบัดทั้งหมด 16 ท่า (side และตำแหน่งร่างกายกำหนดตายตัวตามชุดข้อมูล ไม่ได้เดาจากผลตรวจจับ)

## 1. ค่าเฉลี่ยภาพรวมระดับระบบ (Global Metrics Summary)

* **จำนวนวิดีโอที่ประเมินผลสำเร็จ:** {len(df)} รายการ
* **เวลาทั้งหมดที่ใช้:** {time.time() - start_time:.1f} วินาที

| โมเดล | อัตราการตรวจจับเฉลี่ย (%) | ค่าความคลาดเคลื่อนมุมเฉลี่ยรวม (Global MAE) |
| :--- | :---: | :---: |
| **YOLOv8-pose** | {yolo_global_success:.1f}% | {yolo_global_mae:.2f}° |
| **MoveNet (Thunder)** | {movenet_global_success:.1f}% | {movenet_global_mae:.2f}° |
| **MediaPipe (Heavy)** | {mediapipe_global_success:.1f}% | {mediapipe_global_mae:.2f}° |

---

## 2. ตารางผลการทดสอบแยกตามวิดีโอ (Detailed Video-by-Video Table)

| แหล่งวิดีโอ | ท่าออกกำลังกาย | ข้าง (ตามชุดข้อมูล) | YOLOv8 MAE | MoveNet MAE | MediaPipe MAE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| {joined_rows}

---

## 3. โครงสร้างไฟล์ผลลัพธ์ (Output File Structure)
ผลลัพธ์ได้รับการแยกเก็บเป็นหมวดหมู่ตามหมายเลขโฟลเดอร์ท่าออกกำลังกายดังนี้:
```
{args.output_dir}/
├── batch_evaluation_summary.csv
├── batch_comparison_report.md
├── global_model_comparison.png
└── <EXERCISE_ID>/
    ├── <CAM_ID>_annotated.mp4 (วิดีโอที่พล็อตโครงกระดูกเปรียบเทียบ)
    ├── <CAM_ID>_report.csv (รายงานค่ารายเฟรม)
    ├── <CAM_ID>_report.md (รายงานสรุปของวิดีโอ)
    ├── <CAM_ID>_angle_trajectory.png (กราฟเส้นเปรียบเทียบมุมเคลื่อนไหว)
    └── <CAM_ID>_spatial_error.png (กราฟแท่งความคลาดเคลื่อนพิกเซลข้อต่อ)
```
"""
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"[SUCCESS] Batch Markdown report generated -> {md_report_path}")
    print("\n" + "="*75)
    print("  BATCH PROCESSING COMPLETED SUCCESSFULLY")
    print("="*75)
    print(f"  Total Processed Videos: {len(df)}")
    print(f"  Total Execution Time  : {time.time() - start_time:.1f}s")
    print(f"  Results Directory     : {os.path.abspath(args.output_dir)}")
    print("="*75)

if __name__ == "__main__":
    main()
