"""Streamlit demo for the CS338 traffic monitoring notebook pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
DEFAULT_VEHICLE_MODEL = ROOT / "yolov8n-seg.pt"
DEFAULT_TRACKER = ROOT / "configs" / "botsort.yaml"
RUNS_DIR = ROOT / "outputs" / "streamlit_runs"


@dataclass
class Track:
    track_id: int
    class_id: int
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: Optional[np.ndarray]
    contact_point: Optional[tuple[int, int]] = None


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def load_yolo_model(model_path: str):
    from ultralytics import YOLO

    return YOLO(model_path)


def safe_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def resize_to_width(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width == target_width:
        return frame
    scale = target_width / float(width)
    return cv2.resize(frame, (target_width, int(round(height * scale))), interpolation=cv2.INTER_LINEAR)


def heuristic_road_mask(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array(
        [
            [int(width * 0.10), height - 1],
            [int(width * 0.90), height - 1],
            [int(width * 0.70), int(height * 0.45)],
            [int(width * 0.30), int(height * 0.45)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [points], 1)
    return mask


def extract_merged_mask(result: Any, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = frame_shape
    if isinstance(result, list) and result:
        result = result[0]

    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return np.zeros((height, width), dtype=np.uint8)

    data = safe_to_numpy(masks.data)
    if data.ndim != 3 or data.shape[0] == 0:
        return np.zeros((height, width), dtype=np.uint8)

    merged = np.any(data > 0.5, axis=0).astype(np.uint8)
    return cv2.resize(merged, (width, height), interpolation=cv2.INTER_NEAREST)


def infer_road_mask(
    frame: np.ndarray,
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
) -> np.ndarray:
    if not model_path:
        return heuristic_road_mask(frame)

    model_file = Path(model_path).expanduser()
    if not model_file.exists():
        st.warning("Road model path không tồn tại, app đang dùng road ROI heuristic.")
        return heuristic_road_mask(frame)

    model = load_yolo_model(str(model_file))
    try:
        result = model.predict(source=frame, conf=conf, imgsz=imgsz, device=device, verbose=False)
        mask = extract_merged_mask(result, frame.shape[:2])
        return mask if int(mask.sum()) > 0 else heuristic_road_mask(frame)
    except Exception as exc:
        st.warning(f"Road inference lỗi, app đang dùng road ROI heuristic: {exc}")
        return heuristic_road_mask(frame)


def get_ground_contact(
    bbox: tuple[float, float, float, float],
    shadow_offset: float,
) -> tuple[int, int]:
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0
    y_bottom = y_max - shadow_offset * (y_max - y_min)
    return int(round(x_center)), int(round(y_bottom))


def is_on_road(contact_point: tuple[int, int], road_mask: np.ndarray) -> bool:
    x, y = contact_point
    if 0 <= y < road_mask.shape[0] and 0 <= x < road_mask.shape[1]:
        return bool(road_mask[y, x] == 1)
    return False


def bbox_mask(frame_shape: tuple[int, int], bbox: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    mask[y1:y2, x1:x2] = 1
    return mask


def parse_tracks(result: Any, frame_shape: tuple[int, int]) -> list[Track]:
    if isinstance(result, list) and result:
        result = result[0]

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    bboxes = safe_to_numpy(boxes.xyxy) if hasattr(boxes, "xyxy") else np.empty((0, 4))
    ids = safe_to_numpy(boxes.id) if getattr(boxes, "id", None) is not None else np.arange(len(bboxes))
    confs = safe_to_numpy(boxes.conf) if hasattr(boxes, "conf") else np.ones(len(bboxes), dtype=np.float32)
    classes = safe_to_numpy(boxes.cls) if hasattr(boxes, "cls") else np.zeros(len(bboxes), dtype=np.float32)

    masks = None
    if getattr(result, "masks", None) is not None and getattr(result.masks, "data", None) is not None:
        masks = safe_to_numpy(result.masks.data)

    tracks: list[Track] = []
    for index, bbox in enumerate(bboxes):
        seg_mask = None
        if masks is not None and index < masks.shape[0]:
            seg_mask = (masks[index] > 0.5).astype(np.uint8)
            seg_mask = cv2.resize(seg_mask, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)

        tracks.append(
            Track(
                track_id=int(ids[index]) if index < len(ids) else index,
                class_id=int(classes[index]) if index < len(classes) else 0,
                confidence=float(confs[index]) if index < len(confs) else 1.0,
                bbox=tuple(float(v) for v in bbox),
                mask=seg_mask,
            )
        )
    return tracks


def compute_homography_from_road_mask(road_mask: np.ndarray, bev_size: tuple[int, int]) -> np.ndarray:
    if road_mask is None or int(road_mask.sum()) == 0:
        return np.eye(3, dtype=np.float32)

    contours, _ = cv2.findContours((road_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.eye(3, dtype=np.float32)

    x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
    bev_width, bev_height = bev_size
    src = np.float32([[x, y + height], [x + width, y + height], [x + width, y], [x, y]])
    dst = np.float32([[0, bev_height], [bev_width, bev_height], [bev_width, 0], [0, 0]])
    homography = cv2.findHomography(src, dst)[0]
    return np.eye(3, dtype=np.float32) if homography is None else homography.astype(np.float32)


def load_homography(calibration_path: str) -> Optional[np.ndarray]:
    if not calibration_path:
        return None
    path = Path(calibration_path).expanduser()
    if not path.exists():
        st.warning("Calibration path không tồn tại, app sẽ tự suy homography từ road mask.")
        return None

    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == (3, 3):
        return data.astype(np.float32)
    if isinstance(data, np.ndarray) and data.dtype == object:
        obj = data.item()
        if isinstance(obj, dict) and "H" in obj:
            return np.asarray(obj["H"], dtype=np.float32)
    st.warning("Calibration file không đúng định dạng, app sẽ tự suy homography từ road mask.")
    return None


def apply_ipm(mask: np.ndarray, homography: np.ndarray, bev_size: tuple[int, int]) -> np.ndarray:
    warped = cv2.warpPerspective(mask.astype(np.uint8), homography, bev_size, flags=cv2.INTER_NEAREST)
    return (warped > 0).astype(np.uint8)


def compute_ror(road_bev: np.ndarray, vehicle_bev_masks: list[np.ndarray]) -> float:
    road_area = int(np.sum(road_bev == 1))
    if road_area == 0:
        return 0.0

    occupied_mask = np.zeros_like(road_bev, dtype=np.uint8)
    for vehicle_mask in vehicle_bev_masks:
        occupied_mask = np.logical_or(occupied_mask, vehicle_mask > 0)
    occupied = int(np.sum(occupied_mask & (road_bev == 1)))
    return round((occupied / road_area) * 100.0, 2)


def draw_overlay(
    frame: np.ndarray,
    road_mask: np.ndarray,
    tracks: list[Track],
    accepted_tracks: list[Track],
    ror: float,
) -> np.ndarray:
    canvas = frame.copy()
    road_overlay = np.zeros_like(canvas)
    road_overlay[:, :, 1] = (road_mask * 160).astype(np.uint8)
    canvas = cv2.addWeighted(canvas, 1.0, road_overlay, 0.35, 0)

    accepted_ids = {id(track) for track in accepted_tracks}
    for track in tracks:
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        accepted = id(track) in accepted_ids
        color = (0, 220, 255) if accepted else (90, 90, 90)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track.track_id} C:{track.class_id} {track.confidence:.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if track.contact_point is not None:
            cv2.circle(canvas, track.contact_point, 5, (0, 0, 255), -1)

    cv2.rectangle(canvas, (0, 0), (440, 104), (0, 0, 0), -1)
    cv2.putText(canvas, f"ROR: {ror:.2f}%", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 0), 3)
    cv2.putText(
        canvas,
        f"accepted={len(accepted_tracks)} detected={len(tracks)}",
        (20, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    return canvas


def process_video(
    input_video: Path,
    output_video: Path,
    road_model_path: str,
    vehicle_model_path: str,
    calibration_path: str,
    tracker_path: str,
    max_frames: int,
    road_update_interval: int,
    output_width: int,
    bev_width: int,
    bev_height: int,
    road_conf: float,
    vehicle_conf: float,
    vehicle_iou: float,
    alpha: float,
    imgsz: int,
    device: str,
    progress,
    status,
) -> list[dict[str, Union[float, int]]]:
    vehicle_model = load_yolo_model(vehicle_model_path)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {input_video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frames = min(total_frames, max_frames) if max_frames > 0 and total_frames > 0 else max_frames
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Không đọc được frame đầu tiên từ video.")
    first_frame = resize_to_width(first_frame, output_width)
    height, width = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Không tạo được output video: {output_video}")

    static_homography = load_homography(calibration_path)
    homography = static_homography
    road_mask: Optional[np.ndarray] = None
    rows: list[dict[str, Union[float, int]]] = []
    frame_id = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and frame_id >= max_frames:
                break

            started = time.perf_counter()
            frame = resize_to_width(frame, output_width)

            if frame_id % road_update_interval == 0 or road_mask is None:
                road_mask = infer_road_mask(frame, road_model_path, road_conf, imgsz, device)
                if homography is None:
                    homography = compute_homography_from_road_mask(road_mask, (bev_width, bev_height))

            result = vehicle_model.track(
                source=frame,
                persist=True,
                tracker=tracker_path,
                conf=vehicle_conf,
                iou=vehicle_iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )
            tracks = parse_tracks(result, frame.shape[:2])

            vehicle_bev_masks: list[np.ndarray] = []
            accepted_tracks: list[Track] = []
            assert road_mask is not None
            assert homography is not None
            for track in tracks:
                contact = get_ground_contact(track.bbox, alpha)
                track.contact_point = contact
                if not is_on_road(contact, road_mask):
                    continue
                vehicle_mask = track.mask if track.mask is not None else bbox_mask(frame.shape[:2], track.bbox)
                vehicle_bev_masks.append(apply_ipm(vehicle_mask, homography, (bev_width, bev_height)))
                accepted_tracks.append(track)

            road_bev = apply_ipm(road_mask, homography, (bev_width, bev_height))
            ror = compute_ror(road_bev, vehicle_bev_masks)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            writer.write(draw_overlay(frame, road_mask, tracks, accepted_tracks, ror))
            rows.append(
                {
                    "frame": frame_id,
                    "ror": ror,
                    "detected": len(tracks),
                    "accepted": len(accepted_tracks),
                    "elapsed_ms": round(elapsed_ms, 2),
                }
            )

            frame_id += 1
            denominator = target_frames if target_frames and target_frames > 0 else total_frames
            if denominator:
                progress.progress(min(frame_id / denominator, 1.0))
            status.write(f"Đang xử lý frame {frame_id} | ROR {ror:.2f}% | {elapsed_ms:.1f}ms")
    finally:
        cap.release()
        writer.release()

    progress.progress(1.0)
    return rows


def save_uploaded_video(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    run_dir = RUNS_DIR / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / f"input{suffix}"
    video_path.write_bytes(uploaded_file.getbuffer())
    return video_path


def render_sidebar() -> dict[str, Any]:
    st.sidebar.header("Cấu hình")
    road_model_path = st.sidebar.text_input("Road model path (để trống = heuristic ROI)", value="")
    vehicle_model_path = st.sidebar.text_input(
        "Vehicle model path",
        value=str(DEFAULT_VEHICLE_MODEL if DEFAULT_VEHICLE_MODEL.exists() else ""),
    )
    calibration_path = st.sidebar.text_input("Calibration .npy path (optional)", value="")
    tracker_path = st.sidebar.text_input(
        "Tracker YAML",
        value=str(DEFAULT_TRACKER if DEFAULT_TRACKER.exists() else "botsort.yaml"),
    )

    st.sidebar.divider()
    max_frames = st.sidebar.slider("Số frame demo", min_value=30, max_value=600, value=180, step=30)
    road_update_interval = st.sidebar.slider("Road update interval", min_value=1, max_value=60, value=15, step=1)
    output_width = st.sidebar.select_slider("Output width", options=[640, 960, 1280, 1600, 1920], value=960)
    imgsz = st.sidebar.select_slider("YOLO imgsz", options=[320, 480, 640, 960, 1280], value=640)
    device_options = ["0", "cpu"] if cuda_available() else ["cpu"]
    device = st.sidebar.selectbox("Device", options=device_options, index=0)

    st.sidebar.divider()
    road_conf = st.sidebar.slider("Road confidence", 0.05, 0.90, 0.25, 0.05)
    vehicle_conf = st.sidebar.slider("Vehicle confidence", 0.05, 0.90, 0.30, 0.05)
    vehicle_iou = st.sidebar.slider("Vehicle IoU", 0.10, 0.90, 0.45, 0.05)
    alpha = st.sidebar.slider("Shadow offset alpha", 0.00, 0.20, 0.07, 0.01)
    bev_width = st.sidebar.number_input("BEV width", min_value=200, max_value=2000, value=600, step=50)
    bev_height = st.sidebar.number_input("BEV height", min_value=200, max_value=3000, value=1000, step=50)

    return {
        "road_model_path": road_model_path.strip(),
        "vehicle_model_path": vehicle_model_path.strip(),
        "calibration_path": calibration_path.strip(),
        "tracker_path": tracker_path.strip(),
        "max_frames": int(max_frames),
        "road_update_interval": int(road_update_interval),
        "output_width": int(output_width),
        "imgsz": int(imgsz),
        "device": device,
        "road_conf": float(road_conf),
        "vehicle_conf": float(vehicle_conf),
        "vehicle_iou": float(vehicle_iou),
        "alpha": float(alpha),
        "bev_width": int(bev_width),
        "bev_height": int(bev_height),
    }


def main() -> None:
    st.set_page_config(page_title="CS338 Traffic Monitoring Demo", layout="wide")
    st.title("CS338 Traffic Monitoring Demo")
    st.caption("Demo pipeline: road mask, vehicle tracking, BEV/IPM fusion, và Road Occupancy Rate.")

    config = render_sidebar()
    uploaded_video = st.file_uploader("Upload video MP4/MOV/AVI", type=["mp4", "mov", "avi", "mkv"])

    if not uploaded_video:
        st.info("Upload một video để chạy demo. Road model có thể để trống để dùng ROI hình thang.")
        return

    if not config["vehicle_model_path"] or not Path(config["vehicle_model_path"]).expanduser().exists():
        st.error("Cần `Vehicle model path` hợp lệ để detect/track xe.")
        return

    if not Path(config["tracker_path"]).expanduser().exists():
        st.error("Cần `Tracker YAML` hợp lệ, ví dụ `configs/botsort.yaml`.")
        return

    input_path = save_uploaded_video(uploaded_video)
    output_path = input_path.parent / "processed_demo.mp4"

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Input")
        st.video(str(input_path))
    with right:
        st.subheader("Output")
        output_slot = st.empty()

    if st.button("Chạy demo", type="primary"):
        progress = st.progress(0.0)
        status = st.empty()
        try:
            rows = process_video(
                input_video=input_path,
                output_video=output_path,
                progress=progress,
                status=status,
                **config,
            )
        except Exception as exc:
            st.exception(exc)
            return

        output_slot.video(str(output_path))
        st.success(f"Đã xử lý {len(rows)} frame. Output: {output_path}")

        if rows:
            df = pd.DataFrame(rows)
            metric_cols = st.columns(4)
            metric_cols[0].metric("ROR trung bình", f"{df['ror'].mean():.2f}%")
            metric_cols[1].metric("ROR cao nhất", f"{df['ror'].max():.2f}%")
            metric_cols[2].metric("Xe accepted/frame", f"{df['accepted'].mean():.2f}")
            metric_cols[3].metric("Latency TB", f"{df['elapsed_ms'].mean():.1f}ms")

            st.line_chart(df.set_index("frame")["ror"])
            st.dataframe(df, use_container_width=True)

            jsonl = "\n".join(json.dumps(row) for row in rows)
            st.download_button("Tải ROR JSONL", jsonl, file_name="ror_frames.jsonl", mime="application/jsonl")
            st.download_button(
                "Tải video output",
                output_path.read_bytes(),
                file_name="processed_demo.mp4",
                mime="video/mp4",
            )


if __name__ == "__main__":
    main()
