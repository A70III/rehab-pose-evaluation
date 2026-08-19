# Pose Estimation Comparison for UCO Rehabilitation Dataset

โปรเจกต์นี้ใช้ประเมินและเปรียบเทียบโมเดล pose estimation 3 ตัว ได้แก่ YOLOv8, MoveNet และ MediaPipe กับ ground truth ของชุดข้อมูล UCO Physical Rehabilitation

## Files Included

- `batch_pose_eval.py`  
  โค้ดหลักสำหรับรันประเมินหลายวิดีโอแบบ batch

- `run_comparison_pipeline.py`  
  โค้ดสำหรับรันวิดีโอเดียว ใช้ทดสอบ pipeline หรือทำ demo

- `rebuild_summary.py`  
  ใช้สร้างไฟล์ summary ใหม่จาก report ที่มีอยู่แล้ว โดยไม่ต้องรันโมเดลใหม่

- `model_performance_summary.py`  
  ใช้สรุป performance จาก `output_comparison_results/batch_evaluation_summary.csv`

- `eda_analysis.py`  
  ใช้ทำ exploratory data analysis

- `requirements.txt`  
  รายชื่อ Python packages ที่ต้องติดตั้ง

- `clips_mp4/`  
  โฟลเดอร์ dataset input ประกอบด้วยวิดีโอและ ground truth

## Dataset Structure

โครงสร้าง input หลักควรเป็นแบบนี้:

```text
clips_mp4/0/
├── 01/
│   ├── cam0.mp4
│   ├── cam0_p2d.txt
│   ├── cam1.mp4
│   └── cam1_p2d.txt
├── 02/
│   └── ...
└── 15/
    └── ...
```

แต่ละวิดีโอ `camX.mp4` ควรมี ground truth คู่กันชื่อ `camX_p2d.txt`

## Required Models

ค่า default ในโค้ดอ้างถึงโมเดลตาม path นี้:

```text
c:\Homework\LAB\YOLO\yolov8n-pose.pt
C:\Users\LOQ\AppData\Local\Temp\tfhub_modules\f729a5f3231391676ca61cc7ab789993549d8bca
c:\Homework\LAB\Medie\pose_landmarker_heavy.task
```

ถ้าใช้เครื่องอื่นและเก็บโมเดลคนละ path ให้ระบุ path ใหม่ด้วย argument:

- `--yolo-model`
- `--movenet-model`
- `--mediapipe-model`

## Setup

แนะนำใช้ Python 3.11

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

ถ้า PowerShell ไม่ให้ activate environment ให้รัน:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run Batch Evaluation

รันทั้ง dataset:

```powershell
python batch_pose_eval.py --dir clips_mp4/0 --output-dir output_comparison_results
```

ผลลัพธ์จะถูกสร้างใน:

```text
output_comparison_results/
```

## Quick Test Run

จำกัดจำนวนเฟรมต่อวิดีโอ เช่น 30 เฟรมแรก:

```powershell
python batch_pose_eval.py --dir clips_mp4/0 --output-dir output_comparison_results_test --limit-frames 30
```

ไม่สร้างวิดีโอ annotated เพื่อลดเวลาและขนาดไฟล์:

```powershell
python batch_pose_eval.py --dir clips_mp4/0 --output-dir output_comparison_results_test --limit-frames 30 --no-videos
```

ไม่สร้างกราฟรายวิดีโอ:

```powershell
python batch_pose_eval.py --dir clips_mp4/0 --output-dir output_comparison_results_test --limit-frames 30 --no-plots
```

## CPU or GPU

บังคับใช้ CPU:

```powershell
python batch_pose_eval.py --dir clips_mp4/0 --output-dir output_comparison_results --device cpu
```

บังคับใช้ GPU:

```powershell
python batch_pose_eval.py --dir clips_mp4/0 --output-dir output_comparison_results --device gpu
```

ถ้าไม่ใส่ `--device` โค้ดจะตรวจจับให้อัตโนมัติ

## Supine Rotation

ค่า default จะ rotate exercise ID `3,4,7,8` ก่อน inference เพราะเป็นท่าที่นอนอยู่

กำหนดเอง:

```powershell
python batch_pose_eval.py --dir clips_mp4/0 --output-dir output_comparison_results --rotate-ids 3,4,7,8
```

ปิด rotation ทั้งหมดเพื่อทดสอบ A/B:

```powershell
python batch_pose_eval.py --dir clips_mp4/0 --output-dir output_comparison_results_no_rotate --rotate-ids ""
```

## Run Single Video

ใช้ `run_comparison_pipeline.py` เมื่อต้องการรันวิดีโอเดียว:

```powershell
python run_comparison_pipeline.py --video clips_mp4/0/01/cam1.mp4 --gt clips_mp4/0/01/cam1_p2d.txt --output-video output_pose_comparison.mp4 --output-csv comparison_report_all.csv
```

ถ้าโมเดลอยู่คนละ path:

```powershell
python run_comparison_pipeline.py --video clips_mp4/0/01/cam1.mp4 --gt clips_mp4/0/01/cam1_p2d.txt --yolo-model "PATH\TO\yolov8n-pose.pt" --movenet-model "PATH\TO\movenet_model_folder" --mediapipe-model "PATH\TO\pose_landmarker_heavy.task"
```

## Rebuild Summary Without Running Models

ใช้เมื่อมีไฟล์ `camX_report.csv` อยู่แล้วใน `output_comparison_results/`

```powershell
python rebuild_summary.py
```

## Outputs

ใน `output_comparison_results/` จะมีไฟล์หลัก ๆ ดังนี้:

- `batch_evaluation_summary.csv`: ตารางสรุป metric ของทุกวิดีโอ
- `batch_comparison_report.md`: รายงานสรุปแบบอ่านง่าย
- `global_model_comparison.png`: กราฟเปรียบเทียบภาพรวมของ YOLOv8, MoveNet และ MediaPipe
- `<exercise>/camX_report.csv`: metric รายเฟรมของแต่ละวิดีโอ
- `<exercise>/camX_report.md`: สรุปผลรายวิดีโอแบบ Markdown
- `<exercise>/camX_angle_trajectory.png`: กราฟมุมข้อต่อเทียบกับ ground truth
- `<exercise>/camX_spatial_error.png`: กราฟ error ของตำแหน่งข้อต่อ
- `<exercise>/camX_annotated.mp4`: วิดีโอ overlay skeleton และมุมข้อต่อ
