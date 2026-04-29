"""OpenCV detection demo that saves annotated results.

The script accepts an image path, image URL, video file, or webcam input,
runs object or face detection, and saves:

- `outputs/annotated.jpg`: annotated image for image inputs
- `outputs/annotated.mp4`: annotated video for video/webcam inputs
- `outputs/detections.json`: structured detection results

The model files are downloaded automatically on first run if they are missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable, cast
from urllib.request import Request, urlopen

import cv2
import numpy as np


MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
FACE_CASCADE_NAME = "haarcascade_frontalface_default.xml"

PROTO_TXT_URL = (
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
)
MODEL_URL = (
    "http://dl.caffe.berkeleyvision.org/MobileNetSSD_deploy.caffemodel"
)

PROTO_TXT_NAME = "deploy.prototxt"
MODEL_NAME = "mobilenet_ssd.caffemodel"

# MobileNet-SSD class labels.
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())
    return destination


def ensure_model_files() -> tuple[Path, Path]:
    proto_path = MODEL_DIR / PROTO_TXT_NAME
    model_path = MODEL_DIR / MODEL_NAME

    if not proto_path.exists():
        download_file(PROTO_TXT_URL, proto_path)

    if not model_path.exists():
        download_file(MODEL_URL, model_path)

    return proto_path, model_path


def load_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / FACE_CASCADE_NAME
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load face cascade: {cascade_path}")
    return cascade


def load_image(source: str) -> np.ndarray:
    if source.startswith("http://") or source.startswith("https://"):
        request = Request(source, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=60) as response:
            data = np.frombuffer(response.read(), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode image from URL: {source}")
        return image

    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source}")
    return image


def detect_objects(
    image: np.ndarray,
    proto_path: Path,
    model_path: Path,
    confidence_threshold: float = 0.5,
) -> list[dict[str, object]]:
    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    return detect_objects_with_net(image, net, confidence_threshold)


def detect_objects_with_net(
    image: np.ndarray,
    net: cv2.dnn_Net,
    confidence_threshold: float = 0.5,
) -> list[dict[str, object]]:
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5,
    )
    net.setInput(blob)
    detections = net.forward()

    results: list[dict[str, object]] = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        class_id = int(detections[0, 0, i, 1])
        if class_id >= len(CLASSES):
            label = f"class_{class_id}"
        else:
            label = CLASSES[class_id]

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        start_x, start_y, end_x, end_y = box.astype("int")

        results.append(
            {
                "label": label,
                "confidence": round(confidence, 4),
                "box": [int(start_x), int(start_y), int(end_x), int(end_y)],
            }
        )

    return results


def detect_faces(image: np.ndarray, cascade: cv2.CascadeClassifier) -> list[dict[str, object]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results: list[dict[str, object]] = []
    for (x, y, w, h) in faces:
        results.append(
            {
                "label": "face",
                "confidence": 1.0,
                "box": [int(x), int(y), int(x + w), int(y + h)],
            }
        )
    return results


def draw_detections(image: np.ndarray, detections: Iterable[dict[str, object]]) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = cast(list[int], detection["box"])
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))
        label = str(detection["label"])
        confidence = float(detection["confidence"])
        caption = f"{label}: {confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = max(y1 - 10, 20)
        cv2.putText(
            annotated,
            caption,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def overlay_status(image: np.ndarray, status_lines: list[str]) -> np.ndarray:
    annotated = image.copy()
    x = 10
    y = 28
    for line in status_lines:
        cv2.putText(
            annotated,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 26
    return annotated


def should_write_json(export_format: str) -> bool:
    return export_format in {"json", "both"}


def should_write_csv(export_format: str) -> bool:
    return export_format in {"csv", "both"}


def write_csv_report(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source",
        "media_type",
        "frame_index",
        "label",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_video_input(source: str) -> bool:
    source_path = Path(source)
    return source_path.suffix.lower() in VIDEO_EXTENSIONS


def open_capture(source: str) -> cv2.VideoCapture:
    if source == "0" or source.lower() == "webcam":
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(source)


def process_video(
    source: str,
    mode: str,
    proto_path: Path | None,
    model_path: Path | None,
    output_dir: Path,
    confidence_threshold: float,
    export_format: str,
    display: bool = False,
) -> list[Path]:
    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    net = None
    cascade = None
    if mode == "object":
        if proto_path is None or model_path is None:
            raise RuntimeError("Object mode requires model paths")
        net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    else:
        cascade = load_face_cascade()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    initial_frame: np.ndarray | None = None
    if width <= 0 or height <= 0:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Could not read a frame from: {source}")
        height, width = frame.shape[:2]
        initial_frame = frame

    output_video_path = output_dir / "annotated.mp4"
    video_writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frames: list[dict[str, object]] = []
    csv_rows: list[dict[str, object]] = []
    frame_index = 0
    started_at = time.perf_counter()
    try:
        while True:
            if initial_frame is not None:
                frame = initial_frame
                initial_frame = None
                ok = True
            else:
                ok, frame = cap.read()
            if not ok:
                break

            if mode == "object":
                assert net is not None
                detections = detect_objects_with_net(frame, net, confidence_threshold)
            else:
                assert cascade is not None
                detections = detect_faces(frame, cascade)
            annotated = draw_detections(frame, detections)
            elapsed = time.perf_counter() - started_at
            avg_fps = (frame_index + 1) / elapsed if elapsed > 0 else 0.0
            annotated = overlay_status(
                annotated,
                [
                    f"Frame: {frame_index}",
                    f"Detections: {len(detections)}",
                    f"FPS: {avg_fps:.2f}",
                ],
            )
            video_writer.write(annotated)

            frames.append(
                {
                    "frame_index": frame_index,
                    "detections": detections,
                }
            )
            for detection in detections:
                x1, y1, x2, y2 = cast(list[int], detection["box"])
                csv_rows.append(
                    {
                        "source": source,
                        "media_type": "webcam"
                        if source == "0" or source.lower() == "webcam"
                        else "video",
                        "frame_index": frame_index,
                        "label": str(detection["label"]),
                        "confidence": float(detection["confidence"]),
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                )

            if display:
                cv2.imshow("OpenCV Object Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_index += 1
    finally:
        cap.release()
        video_writer.release()
        if display:
            cv2.destroyAllWindows()

    saved_paths: list[Path] = [output_video_path]

    json_path = output_dir / "detections.json"
    if should_write_json(export_format):
        json_path.write_text(
            json.dumps(
                {
                    "input": source,
                    "threshold": confidence_threshold,
                    "mode": mode,
                    "type": "video"
                    if source != "0" and source.lower() != "webcam"
                    else "webcam",
                    "fps": fps,
                    "frame_count": len(frames),
                    "frames": frames,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        saved_paths.append(json_path)

    csv_path = output_dir / "detections.csv"
    if should_write_csv(export_format):
        write_csv_report(csv_path, csv_rows)
        saved_paths.append(csv_path)

    avg_fps = (frame_index / (time.perf_counter() - started_at)) if frame_index else 0.0
    print(f"Processed {frame_index} frames at ~{avg_fps:.2f} FPS")
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCV object and face detection demo")
    parser.add_argument(
        "--mode",
        choices=("object", "face"),
        default="object",
        help="Detection mode to use.",
    )
    parser.add_argument(
        "--input",
        default="https://raw.githubusercontent.com/opencv/opencv/master/samples/data/dog.jpg",
        help="Path or URL to an input image or video. Use 0 or webcam for the camera.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show a live preview window while processing video or webcam input.",
    )
    parser.add_argument(
        "--export-format",
        choices=("json", "csv", "both"),
        default=None,
        help="Export detections as JSON, CSV, or both. Defaults to CSV for video/webcam and JSON for images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proto_path: Path | None = None
    model_path: Path | None = None
    if args.mode == "object":
        proto_path, model_path = ensure_model_files()
    if args.input == "0" or args.input.lower() == "webcam" or is_video_input(args.input):
        export_format = args.export_format or "csv"
        saved_paths = process_video(
            args.input,
            args.mode,
            proto_path,
            model_path,
            output_dir,
            args.threshold,
            export_format,
            display=args.display,
        )
        for path in saved_paths:
            print(f"Saved {path}")
        return

    image = load_image(args.input)
    if args.mode == "object":
        if proto_path is None or model_path is None:
            raise RuntimeError("Object mode requires model paths")
        detections = detect_objects(image, proto_path, model_path, args.threshold)
    else:
        detections = detect_faces(image, load_face_cascade())
    annotated = draw_detections(image, detections)
    export_format = args.export_format or "json"

    annotated_path = output_dir / "annotated.jpg"
    detections_json_path = output_dir / "detections.json"
    detections_csv_path = output_dir / "detections.csv"

    cv2.imwrite(str(annotated_path), annotated)
    saved_paths = [annotated_path]

    if should_write_json(export_format):
        detections_json_path.write_text(
            json.dumps(
                {
                    "input": args.input,
                    "threshold": args.threshold,
                    "mode": args.mode,
                    "type": "image",
                    "detections": detections,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        saved_paths.append(detections_json_path)

    if should_write_csv(export_format):
        csv_rows = []
        for detection in detections:
            x1, y1, x2, y2 = cast(list[int], detection["box"])
            csv_rows.append(
                {
                    "source": args.input,
                    "media_type": "image",
                    "frame_index": 0,
                    "label": str(detection["label"]),
                    "confidence": float(detection["confidence"]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
        write_csv_report(detections_csv_path, csv_rows)
        saved_paths.append(detections_csv_path)

    for path in saved_paths:
        print(f"Saved {path}")
    print(f"Detections found: {len(detections)}")


if __name__ == "__main__":
    main()
