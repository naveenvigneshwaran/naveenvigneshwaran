from ultralytics import YOLO
from pyzbar.pyzbar import decode

import cv2
import numpy as np
import time

from picamera2 import Picamera2

# ======================================
# SETTINGS
# ======================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

ALLOWED_QR = "http://bn.m.wikipedia.org"

REAL_QR_WIDTH = 5.0
FOCAL_LENGTH = 700

MAX_ZOOM = 1.8
ZOOM_SPEED = 0.12

SMOOTHING = 0.25
BOX_SCALE = 0.55

LOCK_TIMEOUT = 40

# ======================================
# LOAD YOLO
# ======================================

model = YOLO("yolov8n.pt")

# ======================================
# PICAMERA2 SETUP
# ======================================

picam2 = Picamera2()

camera_config = picam2.create_preview_configuration(
    main={
        "size": (
            FRAME_WIDTH,
            FRAME_HEIGHT
        )
    }
)

picam2.configure(camera_config)

picam2.start()

time.sleep(2)

# ======================================
# VARIABLES
# ======================================

smooth_x = FRAME_WIDTH // 2
smooth_y = FRAME_HEIGHT // 2

zoom_level = 1.0

prev_time = time.time()

locked_qr = None
lost_counter = 0

# ======================================
# MAIN LOOP
# ======================================

while True:

    # ======================================
    # CAMERA FRAME
    # ======================================

    frame = picam2.capture_array()

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    original = frame.copy()

    # ======================================
    # FPS
    # ======================================

    current_time = time.time()

    fps = 1 / max(
        current_time - prev_time,
        0.001
    )

    prev_time = current_time

    # ======================================
    # IMAGE PROCESSING
    # ======================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # ======================================
    # QR DETECTION
    # ======================================

    qr_codes = decode(gray)

    found_target = False

    # ======================================
    # QR PROCESSING
    # ======================================

    for qr in qr_codes:

        qr_data = qr.data.decode("utf-8")

        x, y, w, h = qr.rect

        # ======================================
        # LOCK FIRST QR
        # ======================================

        if locked_qr is None:
            locked_qr = qr_data

        if qr_data != locked_qr:
            continue

        found_target = True
        lost_counter = 0

        # ======================================
        # CENTER
        # ======================================

        target_x = x + w // 2
        target_y = y + h // 2

        smooth_x = int(
            smooth_x +
            (target_x - smooth_x)
            * SMOOTHING
        )

        smooth_y = int(
            smooth_y +
            (target_y - smooth_y)
            * SMOOTHING
        )

        # ======================================
        # AUTO ZOOM
        # ======================================

        target_zoom = min(
            MAX_ZOOM,
            max(
                1.0,
                180 / max(w, 60)
            )
        )

        zoom_level += (
            target_zoom - zoom_level
        ) * ZOOM_SPEED

        # ======================================
        # CROP
        # ======================================

        crop_w = int(
            FRAME_WIDTH / zoom_level
        )

        crop_h = int(
            FRAME_HEIGHT / zoom_level
        )

        sx = max(
            smooth_x - crop_w // 2,
            0
        )

        sy = max(
            smooth_y - crop_h // 2,
            0
        )

        ex = min(
            sx + crop_w,
            FRAME_WIDTH
        )

        ey = min(
            sy + crop_h,
            FRAME_HEIGHT
        )

        cropped = original[
            sy:ey,
            sx:ex
        ]

        frame = cv2.resize(
            cropped,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            )
        )

        # ======================================
        # BOX RECALCULATION
        # ======================================

        scale_x = FRAME_WIDTH / crop_w
        scale_y = FRAME_HEIGHT / crop_h

        nx = int((x - sx) * scale_x)
        ny = int((y - sy) * scale_y)

        nw = int(w * scale_x)
        nh = int(h * scale_y)

        rw = int(nw * BOX_SCALE)
        rh = int(nh * BOX_SCALE)

        cx = nx + nw // 2
        cy = ny + nh // 2

        nx = cx - rw // 2
        ny = cy - rh // 2

        # ======================================
        # DISTANCE
        # ======================================

        distance = (
            REAL_QR_WIDTH *
            FOCAL_LENGTH
        ) / max(w, 1)

        # ======================================
        # RANGE
        # ======================================

        if distance < 20:
            range_status = "VERY CLOSE"

        elif distance < 50:
            range_status = "GOOD RANGE"

        elif distance < 100:
            range_status = "MEDIUM RANGE"

        else:
            range_status = "LONG RANGE"

        # ======================================
        # ACCURACY
        # ======================================

        accuracy = min(
            100,
            int((w / 150) * 100)
        )

        # ======================================
        # ACCESS
        # ======================================

        if qr_data == ALLOWED_QR:

            status = "ACCESS ALLOWED"
            color = (0, 255, 0)

        else:

            status = "ACCESS DENIED"
            color = (0, 0, 255)

        # ======================================
        # DRAWING
        # ======================================

        cv2.rectangle(
            frame,
            (nx, ny),
            (nx + rw, ny + rh),
            color,
            2
        )

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (255, 0, 0),
            -1
        )

        cv2.line(
            frame,
            (FRAME_WIDTH // 2, FRAME_HEIGHT // 2),
            (cx, cy),
            (255, 255, 0),
            2
        )

        # ======================================
        # TEXT
        # ======================================

        cv2.putText(
            frame,
            "LOCKED QR",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Accuracy: {accuracy}%",
            (15, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Distance: {distance:.1f} cm",
            (15, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Range: {range_status}",
            (15, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (15, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        break

    # ======================================
    # SEARCH MODE
    # ======================================

    if not found_target:

        lost_counter += 1

        cv2.putText(
            frame,
            "SEARCHING QR...",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        if lost_counter > LOCK_TIMEOUT:

            locked_qr = None
            zoom_level = 1.0

    # ======================================
    # DISPLAY
    # ======================================

    cv2.imshow(
        "ADVANCED QR TRACKING",
        frame
    )

    # ======================================
    # EXIT
    # ======================================

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ======================================
# CLEANUP
# ======================================

picam2.stop()

cv2.destroyAllWindows()