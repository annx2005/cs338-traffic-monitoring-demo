# CS338 Traffic Monitoring Demo

This repository contains a Streamlit demo for the traffic monitoring pipeline from `cs338.ipynb`. The app accepts an uploaded video, runs road segmentation and vehicle tracking, projects the results into Bird's-Eye View (BEV), and computes `Road Occupancy Rate (ROR)` frame by frame.

This repo is focused on demo and inference only. It does not include the full training pipeline, dataset preprocessing flow, or the original Kaggle notebook environment.

## What the demo does

- Upload a video in `mp4`, `mov`, `avi`, or `mkv` format.
- Run a road segmentation model to estimate the drivable area.
- Run YOLO tracking to detect and track vehicles on every frame.
- Filter vehicles with a ground-contact point using `alpha = 0.07`.
- Warp road and vehicle masks into BEV with homography/IPM.
- Compute `ROR = occupied road area / total road area`.
- Export an annotated video, per-frame statistics, a ROR line chart, and a JSONL file.

## Project structure

```text
demo/
├── configs/
│   ├── botsort.yaml
│   └── ocsort.yaml
├── models/
│   ├── reid/
│   │   └── osnet.pt
│   ├── stream1/
│   │   └── road_seg.pt
│   └── stream2/
│       └── vehicle_seg.pt
├── outputs/
│   └── streamlit_runs/
├── .gitignore
├── README.md
├── requirements.txt
└── streamlit_app.py
```

`outputs/` is ignored by Git.

## Requirements

- Python 3.9+
- An environment that can run `ultralytics` and OpenCV
- GPU is optional. The app can run on `cpu`, but it will be much slower.

## Installation

```bash
cd demo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

```bash
cd demo
source .venv/bin/activate
streamlit run streamlit_app.py
```

After the app opens:

1. Upload a video.
2. Review the sidebar settings.
3. Click `Chay demo`.

By default, the app auto-fills these local paths if they exist:

- `models/stream1/road_seg.pt`
- `models/stream2/vehicle_seg.pt`
- `configs/botsort.yaml`

## Main sidebar parameters

- `Road model path`: road segmentation model. If left empty or invalid, the app falls back to a trapezoid ROI heuristic.
- `Vehicle model path`: vehicle detection/segmentation/tracking model. This is required.
- `Calibration .npy path`: a `3x3` homography file. If omitted, the app estimates homography from the road mask.
- `Tracker YAML`: usually `configs/botsort.yaml`.
- `Max frames`: limits the demo length so runs stay manageable.
- `Road update interval`: number of frames between road-mask refreshes.
- `YOLO imgsz`: YOLO inference image size.
- `Device`: `0` for CUDA if available, or `cpu`.

## Where outputs are stored

Each run creates a new timestamped folder:

```text
outputs/streamlit_runs/YYYYMMDD_HHMMSS/
├── input.mp4
└── processed_demo.mp4
```

Inside the app, after processing completes, you can also:

- watch the output video
- download `processed_demo.mp4`
- download `ror_frames.jsonl`

## What the calibration file is

`calibration_1.npy`, or any similar `.npy` file, stores the homography matrix used to transform the camera view into Bird's-Eye View.

This demo repo does not generate calibration files. If you do not provide one, the app still runs, but it uses a homography estimated from the road mask, which is suitable for demo purposes rather than precise measurement.

## Practical notes

- If the input or output video does not play in the browser, the issue is usually codec compatibility, not file corruption.
- Some `.mov` or `.mp4` files may open fine in VLC or QuickTime but still fail inside a browser.
- The current output video is written with `mp4v`, so some browsers may refuse to play it.
- For better browser compatibility, convert the video to H.264 before uploading.

Example with `ffmpeg`:

```bash
ffmpeg -i input.MOV -c:v libx264 -pix_fmt yuv420p -c:a aac input_h264.mp4
```

## Troubleshooting

### `Vehicle model path` is invalid

Make sure this file exists:

```text
models/stream2/vehicle_seg.pt
```

### `proximity_thresh` is missing in BoT-SORT

This repo already includes a patched `configs/botsort.yaml` for the current Ultralytics version. If you replace it with another tracker YAML, make sure it includes:

- `proximity_thresh`
- `appearance_thresh`
- `model`

### Output files are still being added to Git

Make sure `.gitignore` contains:

```gitignore
outputs/
```

## Demo limitations

- No training pipeline is included.
- The app does not currently use Re-ID, even though the repo contains `osnet.pt`.
- The homography fallback is only an approximation.
- ROR values are suitable for demonstrating the architecture, not for official benchmarking.
