"""
rebuild_summary.py
------------------
สร้าง batch_evaluation_summary.csv ใหม่จาก per-video report CSV ทุกไฟล์ที่มีอยู่
โดยไม่ต้อง re-run model ใดๆ

Usage:
    python rebuild_summary.py
"""
import os
import glob
import pandas as pd
import numpy as np

EXERCISE_INFO = {
    1:  ("Bending the knee without support while sitting", "Seated",   "left",  "lower"),
    2:  ("Bending the knee with support while sitting",    "Seated",   "left",  "lower"),
    3:  ("Lift the extended leg",                           "Supine",   "left",  "lower"),
    4:  ("Bending the knee with bed support",               "Supine",   "left",  "lower"),
    5:  ("Bending the knee without support while sitting",  "Seated",   "right", "lower"),
    6:  ("Bending the knee with support while sitting",     "Seated",   "right", "lower"),
    7:  ("Lift the extended leg",                           "Supine",   "right", "lower"),
    8:  ("Bending the knee with bed support",               "Supine",   "right", "lower"),
    9:  ("Shoulder flexion",                                "Seated",   "left",  "upper"),
    10: ("Horizontal weighted openings",                    "Standing", "left",  "upper"),
    11: ("External rotation of shoulders with elastic band","Standing", "left",  "upper"),
    12: ("Circular pendulum",                               "Standing", "left",  "upper"),
    13: ("Shoulder flexion",                                "Seated",   "right", "upper"),
    14: ("Horizontal weighted openings",                    "Standing", "right", "upper"),
    15: ("External rotation of shoulders with elastic band","Standing", "right", "upper"),
    16: ("Circular pendulum",                               "Standing", "right", "upper"),
}
ROTATED_EXERCISE_IDS = {3, 4, 7, 8}

OUTPUT_DIR = r"output_comparison_results"

results = []

# ค้นหา report CSV ทุกไฟล์
report_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*", "*_report.csv")))
print(f"[INFO] พบ {len(report_files)} report CSV files")

for csv_path in report_files:
    folder_name = os.path.basename(os.path.dirname(csv_path))   # e.g. "01"
    video_name  = os.path.basename(csv_path).replace("_report.csv", ".mp4")  # e.g. "cam0.mp4"

    # หา exercise_id จากชื่อโฟลเดอร์
    ex_id = int(folder_name) if folder_name.isdigit() else None
    if ex_id is None or ex_id not in EXERCISE_INFO:
        print(f"  [WARN] ข้ามไฟล์ (ไม่พบ exercise ID): {csv_path}")
        continue

    ex_name, ex_position, ex_side, ex_region = EXERCISE_INFO[ex_id]
    needs_rot = ex_id in ROTATED_EXERCISE_IDS
    is_upper  = (ex_region == "upper")
    j0_name, j1_name, j2_name = ("Shoulder", "Elbow", "Wrist") if is_upper else ("Hip", "Knee", "Ankle")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  [WARN] อ่าน CSV ไม่ได้: {csv_path} -> {e}")
        continue

    num_frames = len(df)
    row = {
        "video":         video_name,
        "exercise":      f"{ex_id:02d}",
        "exercise_name": ex_name,
        "position":      ex_position,
        "side":          ex_side,
        "region":        ex_region,
        "rotated":       needs_rot,
        "folder":        folder_name,
    }

    for model_name, prefix in [("YOLOv8", "yolo"), ("MoveNet", "movenet"), ("MediaPipe", "mediapipe")]:
        angle_err_col = f"{prefix}_angle_err"
        j0_col = f"{prefix}_{j0_name.lower()}_err_px"
        j1_col = f"{prefix}_{j1_name.lower()}_err_px"
        j2_col = f"{prefix}_{j2_name.lower()}_err_px"

        errs = df[angle_err_col].dropna() if angle_err_col in df.columns else pd.Series(dtype=float)
        row[f"{model_name}_valid_pct"] = 100.0 * len(errs) / num_frames if num_frames > 0 else 0.0
        row[f"{model_name}_mae"]       = errs.mean() if len(errs) > 0 else float("nan")
        row[f"{model_name}_j0_err"]    = df[j0_col].dropna().mean() if j0_col in df.columns else float("nan")
        row[f"{model_name}_j1_err"]    = df[j1_col].dropna().mean() if j1_col in df.columns else float("nan")
        row[f"{model_name}_j2_err"]    = df[j2_col].dropna().mean() if j2_col in df.columns else float("nan")

    results.append(row)
    print(f"  [OK] {folder_name}/{video_name} -> {num_frames} frames")

if not results:
    print("[ERROR] ไม่มีข้อมูล")
else:
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUTPUT_DIR, "batch_evaluation_summary.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[SUCCESS] บันทึกสรุปใหม่ {len(results)} แถว -> {out_path}")
    print(out_df[["folder","video","YOLOv8_mae","MoveNet_mae","MediaPipe_mae"]].to_string(index=False))
