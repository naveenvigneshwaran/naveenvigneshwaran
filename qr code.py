#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
from datetime import datetime
import time

import cv2
try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False


def log_detection(value: str, output_path: Path | None) -> None:
    if output_path is None:
        return

    timestamp = datetime.now().isoformat(timespec="seconds")
    with output_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp}\t{value}\n")


def should_log_detection(value: str, last_seen: dict[str, float], cooldown_seconds: float) -> bool:
    now = time.monotonic()
    previous = last_seen.get(value)
    if previous is None or (now - previous) >= cooldown_seconds:
        last_seen[value] = now
        return True
    return False


def detect_from_image(image_path: str) -> int:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not read image: {image_path}", file=sys.stderr)
        return 1

    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image)

    if data:
        print(f"QR code: {data}")
        if points is not None:
            points = points.astype(int)
            # Draw lines around the QR code
            for i in range(len(points[0])):
                pt1 = tuple(points[0][i])
                pt2 = tuple(points[0][(i + 1) % len(points[0])])
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)
            
            # Calculate and draw center circle
            center_x = int(sum(p[0] for p in points[0]) / 4)
            center_y = int(sum(p[1] for p in points[0]) / 4)
            cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)

            x, y = points[0][0]
            cv2.putText(
                image,
                data,
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        cv2.imshow("QR Detector", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    print("No QR code found.")
    cv2.imshow("QR Detector", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 2


def detect_from_camera(
    camera_index: int,
    output_path: Path | None,
    headless: bool,
    cooldown_seconds: float,
) -> int:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: could not open camera {camera_index}", file=sys.stderr)
        return 1

    # Set resolution to 1280x720 as mentioned in the video
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detector = cv2.QRCodeDetector()
    last_seen: dict[str, float] = {}
    prev_time = 0
    print("Press q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Error: could not read frame from camera", file=sys.stderr)
            break

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        data, points, _ = detector.detectAndDecode(frame)
        if data:
            print(f"QR code: {data}")
            if should_log_detection(data, last_seen, cooldown_seconds):
                log_detection(data, output_path)
            if points is not None:
                points = points.astype(int)
                # Draw lines around the QR code
                for i in range(len(points[0])):
                    pt1 = tuple(points[0][i])
                    pt2 = tuple(points[0][(i + 1) % len(points[0])])
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
                
                # Calculate and draw center circle
                center_x = int(sum(p[0] for p in points[0]) / 4)
                center_y = int(sum(p[1] for p in points[0]) / 4)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                x, y = points[0][0]
                cv2.putText(
                    frame,
                    data,
                    (int(x), int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )

        if not headless:
            cv2.imshow("QR Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    return 0


def detect_from_picamera(output_path: Path | None, headless: bool, cooldown_seconds: float) -> int:
    if not PICAM_AVAILABLE:
        print("Error: Picamera2 library is not installed. Use --camera instead.", file=sys.stderr)
        return 1
    
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    detector = cv2.QRCodeDetector()
    last_seen: dict[str, float] = {}
    prev_time = 0
    print("Press q to quit.")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            data, points, _ = detector.detectAndDecode(frame)
            if data:
                print(f"QR code: {data}")
                if should_log_detection(data, last_seen, cooldown_seconds):
                    log_detection(data, output_path)
                if points is not None:
                    points = points.astype(int)
                    # Draw lines around the QR code
                    for i in range(len(points[0])):
                        pt1 = tuple(points[0][i])
                        pt2 = tuple(points[0][(i + 1) % len(points[0])])
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
                    
                    # Calculate and draw center circle
                    center_x = int(sum(p[0] for p in points[0]) / 4)
                    center_y = int(sum(p[1] for p in points[0]) / 4)
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

                    x, y = points[0][0]
                    cv2.putText(
                        frame,
                        data,
                        (int(x), int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            # Display FPS
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            if not headless:
                cv2.imshow("QR Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        picam2.stop()
        if not headless:
            cv2.destroyAllWindows()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="QR code detector for Raspberry Pi 5")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to an image file")
    group.add_argument("--camera", type=int, help="Camera index, usually 0")
    group.add_argument("--picamera", action="store_true", help="Use Raspberry Pi camera module via picamera2")
    parser.add_argument("--output", help="Append detected QR values to this text file")
    parser.add_argument(
        "--cooldown",
        type=float,
        default=10.0,
        help="Seconds to wait before logging the same QR value again",
    )
    parser.add_argument("--headless", action="store_true", help="Do not open a preview window")
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    if args.image:
        return detect_from_image(args.image)
    if args.picamera:
        return detect_from_picamera(output_path, args.headless, args.cooldown)
    return detect_from_camera(args.camera, output_path, args.headless, args.cooldown)


if __name__ == "__main__":
    raise SystemExit(main())
