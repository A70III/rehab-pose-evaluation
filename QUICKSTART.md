# Quick-Start Guide — Multi-Model Pose Evaluation

รันได้ทั้งบน **MacBook Pro M1** และ **RunPod (NVIDIA GPU Cloud)**

---

## 1. Clone Repository

```bash
git clone <YOUR_REPO_URL>
cd dataset-001
```

---

## 2A. MacBook Pro M1 (Apple Silicon)

### Setup Environment (ทำครั้งเดียว)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install ai-edge-litert
```

### รัน — Dataset ใหม่ (Multi-subject, 393 วิดีโอ)
```bash
python evaluate_all_mac.py --dir clips_mp4_cam1
```

### รัน — Dataset เดิม (Single-subject, 61 วิดีโอ)
```bash
python evaluate_all_mac.py --dir clips_mp4/0
```

### รันแบบทดสอบด่วน (5 เฟรมแรก)
```bash
python evaluate_all_mac.py --dir clips_mp4_cam1 --limit-frames 5 --no-plots --output-dir output_test
```

> สคริปต์จะ detect Apple Silicon MPS อัตโนมัติ ไม่ต้องใส่ `--device`

---

## 2B. RunPod (NVIDIA GPU)

### เช่า Pod
1. สมัคร [runpod.io](https://www.runpod.io) เติมเงิน $10
2. **Deploy Pod** → Secure Cloud → GPU: **RTX 4090**
3. Template: `RunPod PyTorch 2.x` (มี CUDA พร้อมแล้ว)
4. Storage: 50 GB → Deploy → Connect → Jupyter Lab / Web Terminal

### อัปโหลด Dataset

**วิธี A — Git**
```bash
git clone <YOUR_REPO_URL>
cd dataset-001
```

**วิธี B — Google Drive**
```bash
pip install gdown
gdown --folder "https://drive.google.com/drive/folders/<FOLDER_ID>" -O clips_mp4_cam1
```

**วิธี C — rsync จาก Mac** (รันบน Mac)
```bash
rsync -avz --progress \
  /Users/l2s/L2S/Research/EvaluationForEnsemble/dataset-001/ \
  root@<RUNPOD_IP>:<PORT>/workspace/dataset-001/ \
  -e "ssh -p <PORT>"
```

### Setup + Run (393 วิดีโอ ใช้เวลา ~15–25 นาที)
```bash
cd dataset-001
pip install -r requirements.txt

python evaluate_all_mac.py \
  --dir clips_mp4_cam1 \
  --output-dir output_comparison_results \
  --no-plots \
  --device 0
```

### ดาวน์โหลด Output กลับ Mac
```bash
rsync -avz --progress \
  root@<RUNPOD_IP>:<PORT>/workspace/dataset-001/output_comparison_results/ \
  /Users/l2s/L2S/Research/EvaluationForEnsemble/dataset-001/output_comparison_results/
```

> **Stop Pod** ทันทีหลังดาวน์โหลดเสร็จ

---

## 3. Arguments Reference

| Argument | Default | ความหมาย |
|:---|:---|:---|
| `--dir` | `clips_mp4_cam1` | Dataset root |
| `--output-dir` | `output_comparison_results` | โฟลเดอร์ output |
| `--limit-frames N` | (ไม่จำกัด) | จำกัดเฟรมต่อวิดีโอ |
| `--no-plots` | False | ปิดกราฟรายวิดีโอ (เร็วขึ้น) |
| `--save-videos` | False | บันทึกวิดีโอ annotated |
| `--device` | (auto) | `mps` / `cpu` / `0` (CUDA) |
| `--rotate-ids` | `3,4,7,8` | Exercise ID ที่หมุนภาพก่อน inference |

---

## 4. Output Structure

```
output_comparison_results/
├── batch_evaluation_summary.csv       ← ผลทุกวิดีโอ (มี subject_id)
├── batch_comparison_report.md
├── model_performance_summary.csv
├── mae_by_exercise.csv
├── detection_rate_by_exercise.csv
├── global_model_comparison.png
├── eda_analysis_plots.png
└── subject_00/
    ├── 01/
    │   ├── cam1_report.csv
    │   ├── cam1_angle_trajectory.png
    │   └── cam1_spatial_error.png
    └── 16/ ...
```

---

## 5. Auto-Resume

กด `Ctrl+C` กลางคันได้ — รันใหม่แล้วสคริปต์จะ **skip วิดีโอที่เสร็จแล้วอัตโนมัติ**

```bash
python evaluate_all_mac.py --dir clips_mp4_cam1   # รันซ้ำได้เลย
```

---

## 6. Expected Runtime

| สภาพแวดล้อม | วิดีโอ | เวลา |
|:---|:---:|:---:|
| MacBook Pro M1 | 393 | ~10–11 ชั่วโมง |
| MacBook Pro M1 `--no-plots` | 393 | ~9 ชั่วโมง |
| RunPod RTX 4090 `--no-plots` | 393 | **~15–25 นาที** |
