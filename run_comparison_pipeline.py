import os
import sys
import argparse
import math
import cv2
import numpy as np
import pandas as pd

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
# Geometry Helpers
# ------------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """
    Compute flexion angle (in degrees) formed by vector segments BA and BC
    using the dot product formula:
       alpha = arccos( (BA . BC) / (|BA| * |BC|) )
    """
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
    # Guard against degenerate keypoints (0,0)
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
    
    # Draw limb lines
    cv2.line(frame, pt0, pt1, color, 2, cv2.LINE_AA)
    cv2.line(frame, pt1, pt2, color, 2, cv2.LINE_AA)
    
    # Draw joint circles
    for pt in (pt0, pt1, pt2):
        cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 7, (255, 255, 255), 1, cv2.LINE_AA) # White ring
        
    # Draw angle text near the middle joint
    if not np.isnan(angle):
        text = f"{label}: {angle:.1f} deg"
        pos = (pt1[0] + 12, pt1[1] + text_offset_y)
        # Drop shadow
        cv2.putText(frame, text, (pos[0]+1, pos[1]+1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

# ------------------------------------------------------------------------------
# Main Pipeline Execution
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8-pose, MoveNet, and MediaPipe against Ground Truth.")
    parser.add_argument("--video", default=r"clips_mp4/0/01/cam1.mp4", help="Path to input video file")
    parser.add_argument("--gt", default=r"clips_mp4/0/01/cam1_p2d.txt", help="Path to ground truth camX_p2d.txt file")
    parser.add_argument("--yolo-model", default=r"c:\Homework\LAB\YOLO\yolov8n-pose.pt", help="Path to YOLOv8-pose model weights")
    parser.add_argument("--movenet-model", default=r"C:\Users\LOQ\AppData\Local\Temp\tfhub_modules\f729a5f3231391676ca61cc7ab789993549d8bca", help="Path to MoveNet SinglePose Thunder cached TFHub module")
    parser.add_argument("--mediapipe-model", default=r"c:\Homework\LAB\Medie\pose_landmarker_heavy.task", help="Path to MediaPipe Pose Landmarker task model")
    parser.add_argument("--output-video", default=r"output_pose_comparison.mp4", help="Path to save annotated output video")
    parser.add_argument("--output-csv", default=r"comparison_report_all.csv", help="Path to save per-frame CSV report")
    args = parser.parse_args()

    # Validate file existences
    for path_name, path in [("Video", args.video), ("Ground Truth", args.gt), 
                            ("YOLO Weights", args.yolo_model), ("MoveNet model directory", args.movenet_model),
                            ("MediaPipe Task", args.mediapipe_model)]:
        if not os.path.exists(path):
            if path_name == "MoveNet model directory":
                print(f"[INFO] MoveNet directory '{path}' not found. Will download from TF Hub.")
                continue
            sys.exit(f"[ERROR] {path_name} file/folder not found: {path}")

    print("\n" + "="*70)
    print("  INITIALIZING MULTI-MODEL PHYSICAL THERAPY POSE EVALUATION PIPELINE")
    print("="*70)

    # 1. Load Ground Truth coordinates (Shoulder/Hip, Elbow/Knee, Wrist/Ankle x/y)
    print("[INFO] Loading ground truth tracking file...")
    gt_coords = []
    with open(args.gt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) >= 6:
                # [joint0_x, joint0_y, joint1_x, joint1_y, joint2_x, joint2_y]
                gt_coords.append({
                    "j0": (parts[0], parts[1]),
                    "j1": (parts[2], parts[3]),
                    "j2": (parts[4], parts[5])
                })
    print(f"[INFO] Loaded {len(gt_coords)} ground truth frames.")

    # 2. Open Video Capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video file: {args.video}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video resolution: {width}x{height} | FPS: {fps:.1f} | Total frames: {total_frames}")

    # Align frame count with ground truth length
    num_frames_to_process = min(total_frames, len(gt_coords))
    print(f"[INFO] Will process first {num_frames_to_process} frames.")

    # 3. Auto-detect upper vs lower body based on exercise ID from video path
    # Path pattern: clips_mp4/<patient_id>/<exercise_id>/cam<N>.mp4
    exercise_id = 1
    try:
        norm_path = os.path.normpath(args.video).replace("\\", "/")
        parts = norm_path.split("/")
        for part in parts:
            if part.isdigit() and 1 <= int(part) <= 16:
                exercise_id = int(part)
                break
    except Exception as e:
        print(f"[WARN] Could not parse exercise ID from path: {e}. Defaulting to exercise 1.")

    is_upper = (exercise_id >= 9)
    if is_upper:
        j0_name, j1_name, j2_name = "Shoulder", "Elbow", "Wrist"
        angle_name = "Elbow Angle"
    else:
        j0_name, j1_name, j2_name = "Hip", "Knee", "Ankle"
        angle_name = "Knee Angle"

    body_part_label = f"Upper Body ({j0_name}-{j1_name}-{j2_name})" if is_upper else f"Lower Body ({j0_name}-{j1_name}-{j2_name})"
    print(f"[INFO] Auto-detected Exercise ID: {exercise_id:02d} -> {body_part_label}")

    # Rotation hack check for supine positions (Exercises 03, 04, 07, 08)
    needs_rotation = exercise_id in [3, 4, 7, 8]
    rotation_type = "90_cw" if needs_rotation else None
    if needs_rotation:
        print("[INFO] Supine exercise detected! Activating 90-degree clockwise rotation hack.")

    # 4. Load Models
    # A. YOLOv8
    print("[INFO] Loading YOLOv8-Pose model...")
    yolo_model = YOLO(args.yolo_model)

    # B. MoveNet
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

    # C. MediaPipe
    print("[INFO] Loading MediaPipe Pose Landmarker...")
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    mp_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.mediapipe_model),
        running_mode=VisionRunningMode.VIDEO
    )
    mp_landmarker = PoseLandmarker.create_from_options(mp_options)

    # 5. Auto-detect which side (Left or Right) matches Ground Truth
    print("[INFO] Determining whether Ground Truth tracks the LEFT or RIGHT side...")
    left_side_errors = []
    right_side_errors = []
    
    # Inspect first 15 frames for side selection
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for check_idx in range(min(15, num_frames_to_process)):
        ret, frame = cap.read()
        if not ret:
            break
            
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
        gt_data = gt_coords[check_idx]
        gt_j0 = gt_data["j0"]
        
        yolo_results = yolo_model(frame, verbose=False)
        if yolo_results and yolo_results[0].keypoints is not None:
            xy_tensors = yolo_results[0].keypoints.xy
            if xy_tensors.shape[0] > 0:
                person_kps = xy_tensors[0].cpu().numpy() # (17, 2)
                
                idx_l = 5 if is_upper else 11
                idx_r = 6 if is_upper else 12
                
                kp_l = person_kps[idx_l]
                kp_r = person_kps[idx_r]
                
                # Map back if rotated
                if needs_rotation:
                    kp_l = map_rotated_kps_to_orig(kp_l[0], kp_l[1], width, height, rotation_type)
                    kp_r = map_rotated_kps_to_orig(kp_r[0], kp_r[1], width, height, rotation_type)
                
                if not (kp_l[0] < 1.0 and kp_l[1] < 1.0):
                    left_side_errors.append(np.linalg.norm(np.array(kp_l) - np.array(gt_j0)))
                if not (kp_r[0] < 1.0 and kp_r[1] < 1.0):
                    right_side_errors.append(np.linalg.norm(np.array(kp_r) - np.array(gt_j0)))
                    
    # Reset video capture to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    mean_l_err = np.mean(left_side_errors) if left_side_errors else float("inf")
    mean_r_err = np.mean(right_side_errors) if right_side_errors else float("inf")
    
    selected_side = "left" if mean_l_err < mean_r_err else "right"
    print(f"[INFO] Initial tracking errors -> Left: {mean_l_err:.2f} px | Right: {mean_r_err:.2f} px")
    print(f"[SUCCESS] Auto-selected side: {selected_side.upper()} (based on lowest proximal joint error)")

    # 6. Set up Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # 7. Define Colors (BGR)
    COLOR_GT = (0, 220, 80)        # Bright Green
    COLOR_YOLO = (0, 0, 255)       # Bright Red
    COLOR_MOVENET = (255, 140, 0)  # Bright Orange/Blue (BGR: Blue-heavy Orange)
    COLOR_MP = (0, 240, 240)       # Bright Yellow/Cyan

    # Data collection list
    records = []

    print("[INFO] Executing multi-model predictions frame by frame...")
    
    # Set indices based on body part and side
    if selected_side == "left":
        idx_s, idx_e, idx_w = (5, 7, 9) if is_upper else (11, 13, 15)
        mp_idx_s, mp_idx_e, mp_idx_w = (11, 13, 15) if is_upper else (23, 25, 27)
    else:
        idx_s, idx_e, idx_w = (6, 8, 10) if is_upper else (12, 14, 16)
        mp_idx_s, mp_idx_e, mp_idx_w = (12, 14, 16) if is_upper else (24, 26, 28)

    for frame_idx in range(num_frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Ground Truth calculations
        gt_data = gt_coords[frame_idx]
        gt_j0 = gt_data["j0"]
        gt_j1 = gt_data["j1"]
        gt_j2 = gt_data["j2"]
        gt_angle = calculate_angle(gt_j0, gt_j1, gt_j2)

        # Prepare frame for model inference (rotate if supine)
        if needs_rotation:
            inf_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            proc_w, proc_h = height, width
        else:
            inf_frame = frame
            proc_w, proc_h = width, height

        # A. YOLOv8-pose prediction
        yolo_results = yolo_model(inf_frame, verbose=False)
        yolo_s = yolo_e = yolo_w = (float("nan"), float("nan"))
        yolo_angle = float("nan")
        
        if yolo_results and yolo_results[0].keypoints is not None:
            xy_tensors = yolo_results[0].keypoints.xy
            if xy_tensors.shape[0] > 0:
                person_kps = xy_tensors[0].cpu().numpy() # (17, 2)
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

        # B. MoveNet prediction
        # MoveNet Thunder input format: [1, 256, 256, 3] tensor of int32
        movenet_resized = cv2.resize(inf_frame, (256, 256))
        movenet_rgb = cv2.cvtColor(movenet_resized, cv2.COLOR_BGR2RGB)
        movenet_input = tf.convert_to_tensor(np.expand_dims(movenet_rgb, axis=0), dtype=tf.int32)
        movenet_outputs = movenet_fn(movenet_input)
        movenet_kps = movenet_outputs['output_0'].numpy()[0, 0] # (17, 3) -> [y, x, score]
        
        movenet_s = movenet_e = movenet_w = (float("nan"), float("nan"))
        movenet_angle = float("nan")
        CONF_THRESH = 0.20
        if (movenet_kps[idx_s, 2] >= CONF_THRESH and 
            movenet_kps[idx_e, 2] >= CONF_THRESH and 
            movenet_kps[idx_w, 2] >= CONF_THRESH):
            # Convert normalized y, x to absolute x, y pixels in inference space
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

        # C. MediaPipe prediction
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=inf_frame)
        # Use frame index based monotonic timestamp to prevent Windows capture timestamp duplication
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
                
            # Filter low visibility landmarks
            if (landmarks[mp_idx_s].visibility > 0.1 and 
                landmarks[mp_idx_e].visibility > 0.1 and 
                landmarks[mp_idx_w].visibility > 0.1):
                mp_angle = calculate_angle(mp_s, mp_e, mp_w)
            else:
                mp_s = mp_e = mp_w = (float("nan"), float("nan"))

        # D. Calculate errors relative to Ground Truth
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

        # E. Draw overlays on the frame
        draw_pose(frame, gt_j0, gt_j1, gt_j2, gt_angle, COLOR_GT, "GT", text_offset_y=-30)
        draw_pose(frame, yolo_s, yolo_e, yolo_w, yolo_angle, COLOR_YOLO, "YOLO", text_offset_y=-10)
        draw_pose(frame, movenet_s, movenet_e, movenet_w, movenet_angle, COLOR_MOVENET, "MN", text_offset_y=10)
        draw_pose(frame, mp_s, mp_e, mp_w, mp_angle, COLOR_MP, "MP", text_offset_y=30)

        # Draw HUD Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (330, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        def draw_hud_text(text, y_pos, color=(255,255,255), scale=0.45):
            cv2.putText(frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

        draw_hud_text(f"Frame: {frame_idx + 1} / {num_frames_to_process} ({selected_side.upper()} side)", 30, scale=0.5, color=(255,255,255))
        draw_hud_text(f"GT {angle_name}  : {gt_angle:.1f} deg" if not np.isnan(gt_angle) else f"GT {angle_name}  : N/A", 55, COLOR_GT)
        draw_hud_text(f"YOLO Angle (Err): {yolo_angle:.1f} deg ({yolo_angle_err:.1f})" if not np.isnan(yolo_angle) else "YOLO Angle : N/A", 75, COLOR_YOLO)
        draw_hud_text(f"MoveNet (Err)   : {movenet_angle:.1f} deg ({movenet_angle_err:.1f})" if not np.isnan(movenet_angle) else "MoveNet : N/A", 95, COLOR_MOVENET)
        draw_hud_text(f"MediaPipe (Err) : {mp_angle:.1f} deg ({mp_angle_err:.1f})" if not np.isnan(mp_angle) else "MediaPipe : N/A", 115, COLOR_MP)

        # Draw Legend in bottom right
        cv2.rectangle(frame, (width - 180, height - 100), (width - 10, height - 10), (30,30,30), -1)
        def draw_legend(text, y_pos, color):
            cv2.circle(frame, (width - 165, y_pos - 5), 4, color, -1)
            cv2.putText(frame, text, (width - 150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
            
        draw_legend("OptiTrack (GT)", height - 80, COLOR_GT)
        draw_legend("YOLOv8-Pose", height - 60, COLOR_YOLO)
        draw_legend("MoveNet (Thunder)", height - 40, COLOR_MOVENET)
        draw_legend("MediaPipe (Heavy)", height - 20, COLOR_MP)

        writer.write(frame)

        # Record metrics
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
            f"mediapipe_{j2_name.lower()}_err_px": round(mp_j2_err, 3) if not np.isnan(mp_j2_err) else None
        })

        if (frame_idx + 1) % 50 == 0:
            print(f"  Processed {frame_idx + 1}/{num_frames_to_process} frames...")

    cap.release()
    writer.release()
    mp_landmarker.close()
    
    # Save CSV
    df = pd.DataFrame(records)
    df.to_csv(args.output_csv, index=False)
    print(f"[SUCCESS] Annotated video saved -> {args.output_video}")
    print(f"[SUCCESS] CSV metrics report saved -> {args.output_csv}")

    # Compute Summary statistics
    y_angle_errs = df["yolo_angle_err"].dropna()
    mn_angle_errs = df["movenet_angle_err"].dropna()
    mp_angle_errs = df["mediapipe_angle_err"].dropna()

    summary_stats = {
        "YOLOv8": {
            "mae": y_angle_errs.mean(),
            "median": y_angle_errs.median(),
            "max": y_angle_errs.max(),
            "j0_err": df[f"yolo_{j0_name.lower()}_err_px"].dropna().mean(),
            "j1_err": df[f"yolo_{j1_name.lower()}_err_px"].dropna().mean(),
            "j2_err": df[f"yolo_{j2_name.lower()}_err_px"].dropna().mean(),
            "valid_pct": 100.0 * len(y_angle_errs) / num_frames_to_process
        },
        "MoveNet": {
            "mae": mn_angle_errs.mean(),
            "median": mn_angle_errs.median(),
            "max": mn_angle_errs.max(),
            "j0_err": df[f"movenet_{j0_name.lower()}_err_px"].dropna().mean(),
            "j1_err": df[f"movenet_{j1_name.lower()}_err_px"].dropna().mean(),
            "j2_err": df[f"movenet_{j2_name.lower()}_err_px"].dropna().mean(),
            "valid_pct": 100.0 * len(mn_angle_errs) / num_frames_to_process
        },
        "MediaPipe": {
            "mae": mp_angle_errs.mean(),
            "median": mp_angle_errs.median(),
            "max": mp_angle_errs.max(),
            "j0_err": df[f"mediapipe_{j0_name.lower()}_err_px"].dropna().mean(),
            "j1_err": df[f"mediapipe_{j1_name.lower()}_err_px"].dropna().mean(),
            "j2_err": df[f"mediapipe_{j2_name.lower()}_err_px"].dropna().mean(),
            "valid_pct": 100.0 * len(mp_angle_errs) / num_frames_to_process
        }
    }

    # Print summary console report
    print("\n" + "="*70)
    print("  POSE ESTIMATION BENCHMARK EVALUATION SUMMARY")
    print("="*70)
    print(f"  Processed frames : {num_frames_to_process}")
    print(f"  Detected Side    : {selected_side.upper()}")
    print(f"  Joints Evaluated : {j0_name} -> {j1_name} -> {j2_name}")
    for model_name, stats in summary_stats.items():
        print(f"\n  -- {model_name} --")
        print(f"    Detection Success Rate   : {stats['valid_pct']:.1f}%")
        print(f"    Mean Angle Error (MAE)   : {stats['mae']:.2f}°" if not np.isnan(stats['mae']) else "    Mean Angle Error (MAE)   : NaN")
        print(f"    Median Angle Error       : {stats['median']:.2f}°" if not np.isnan(stats['median']) else "    Median Angle Error       : NaN")
        print(f"    Max Angle Error          : {stats['max']:.2f}°" if not np.isnan(stats['max']) else "    Max Angle Error          : NaN")
        print(f"    Mean Spatial Joint Errors:")
        print(f"      - {j0_name} (Proximal)  : {stats['j0_err']:.2f} px" if not np.isnan(stats['j0_err']) else f"      - {j0_name} (Proximal)  : NaN")
        print(f"      - {j1_name} (Mid)       : {stats['j1_err']:.2f} px" if not np.isnan(stats['j1_err']) else f"      - {j1_name} (Mid)       : NaN")
        print(f"      - {j2_name} (Distal)    : {stats['j2_err']:.2f} px" if not np.isnan(stats['j2_err']) else f"      - {j2_name} (Distal)    : NaN")
    print("="*70 + "\n")

    # Save Markdown report in Thai
    md_report_path = "model_comparison_three_way.md"
    md_content = f"""# รายงานเปรียบเทียบประสิทธิภาพโมเดลตรวจจับท่าทาง (YOLOv8, MoveNet, MediaPipe)

การเปรียบเทียบประสิทธิภาพของโมเดล **YOLOv8-pose**, **MoveNet SinglePose Thunder**, และ **MediaPipe (Pose Landmarker Heavy)** 
โดยอ้างอิงจากมุมการเคลื่อนไหวข้อต่อ **{angle_name}** และพิกัดตำแหน่งข้อต่อสี่จุดหลัก (**{j0_name}**, **{j1_name}**, **{j2_name}**) เทียบกับ **OptiTrack Ground Truth**
จากชุดข้อมูลวิดีโอบำบัดทางกายภาพ (`{args.video}`)

## 1. ผลลัพธ์การวัดผลเชิงสถิติ (Kinematic Angle and Spatial Coordinate Errors)

* **โมเดลที่ใช้วิเคราะห์:** {body_part_label}
* **ข้างที่ทำการประเมิน:** {selected_side.upper()} (ระบบตรวจพบข้างที่ถูกต้องโดยอัตโนมัติ)
* **จำนวนเฟรมที่ประมวลผล:** {num_frames_to_process} เฟรม

| โมเดล | อัตราตรวจจับสำเร็จ (%) | ค่าความคลาดเคลื่อนมุมเฉลี่ย (MAE) | ค่ามัธยฐานคลาดเคลื่อนมุม | ค่าคลาดเคลื่อนมุมสูงสุด | {j0_name} Error (px) | {j1_name} Error (px) | {j2_name} Error (px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv8-pose** | {summary_stats['YOLOv8']['valid_pct']:.1f}% | {summary_stats['YOLOv8']['mae']:.2f}° | {summary_stats['YOLOv8']['median']:.2f}° | {summary_stats['YOLOv8']['max']:.2f}° | {summary_stats['YOLOv8']['j0_err']:.2f} px | {summary_stats['YOLOv8']['j1_err']:.2f} px | {summary_stats['YOLOv8']['j2_err']:.2f} px |
| **MoveNet (Thunder)** | {summary_stats['MoveNet']['valid_pct']:.1f}% | {summary_stats['MoveNet']['mae']:.2f}° | {summary_stats['MoveNet']['median']:.2f}° | {summary_stats['MoveNet']['max']:.2f}° | {summary_stats['MoveNet']['j0_err']:.2f} px | {summary_stats['MoveNet']['j1_err']:.2f} px | {summary_stats['MoveNet']['j2_err']:.2f} px |
| **MediaPipe (Heavy)** | {summary_stats['MediaPipe']['valid_pct']:.1f}% | {summary_stats['MediaPipe']['mae']:.2f}° | {summary_stats['MediaPipe']['median']:.2f}° | {summary_stats['MediaPipe']['max']:.2f}° | {summary_stats['MediaPipe']['j0_err']:.2f} px | {summary_stats['MediaPipe']['j1_err']:.2f} px | {summary_stats['MediaPipe']['j2_err']:.2f} px |

*หมายเหตุ: ค่าความคลาดเคลื่อนเป็น NaN แสดงว่าโมเดลไม่มีการตรวจจับหรือถูกคัดกรองออกเนื่องจากค่าความเชื่อมั่นต่ำ*

## 2. บทวิเคราะห์หลักและข้อสังเกตเชิงชีวกลศาสตร์ (Key Insights & Biomechanical Observations)

1. **ความถูกต้องในการคำนวณองศาข้อต่อ (Angle Accuracy):**
   - เปรียบเทียบความถูกต้องของแต่ละโมเดลในการวัดมุมข้อต่อสำคัญระหว่างกายภาพบำบัด ซึ่งมีความจำเป็นมากต่อการประเมินช่วงมุมเคลื่อนไหวข้อต่อ (Range of Motion - ROM) ของคนไข้

2. **ความคลาดเคลื่อนข้อต่อส่วนต้นเทียบกับส่วนปลาย (Proximal vs. Distal Stability):**
   - ข้อต่อส่วนต้นลำตัว (**{j0_name} - Proximal**) มักมีความแม่นยำสูงกว่าและสั่นไหวน้อยกว่าเมื่อเทียบกับข้อต่อส่วนปลายที่เคลื่อนที่อิสระมากกว่า (**{j2_name} - Distal**) การเปรียบเทียบนี้ช่วยยืนยันโครงสร้างความคลาดเคลื่อน (Spatial Error Propagation) ของโมเดลแต่ละแบบ

3. **ผลลัพธ์วิดีโอ (Annotated Video Representation):**
   - วิดีโอผลลัพธ์ได้รับการบันทึกไว้ที่ `{args.output_video}` ซึ่งมีการซ้อนทับภาพโครงกระดูก (Skeletons) และค่ามุมข้อศอก/ข้อเข่าของแต่ละโมเดลด้วยสีเฉพาะตัว เพื่อเปรียบเทียบความคลาดเคลื่อนเชิงสายตาอย่างชัดเจน
"""
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[SUCCESS] Markdown report generated -> {md_report_path}")

if __name__ == "__main__":
    main()
