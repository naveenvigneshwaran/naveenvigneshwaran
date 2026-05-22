from pyzbar.pyzbar import decode
from picamera2 import Picamera2

import cv2
import numpy as np
import time

# ==========================================
# PERFORMANCE SETTINGS
# ==========================================

cv2.setUseOptimized(True)

# ==========================================
# CAMERA SETTINGS
# ==========================================

FRAME_WIDTH = 960
FRAME_HEIGHT = 540

FPS_LIMIT = 30

# ==========================================
# QR SETTINGS
# ==========================================

ALLOWED_QR = "http://bn.m.wikipedia.org"

REAL_QR_WIDTH = 5.0
FOCAL_LENGTH = 950

# ==========================================
# ZOOM SETTINGS
# ==========================================

MIN_ZOOM = 1.0
MAX_ZOOM = 3.5

ZOOM_SPEED = 0.08
SMOOTHING = 0.12

DEAD_ZONE = 20

BOX_SCALE = 0.75

# ==========================================
# DETECTION SETTINGS
# ==========================================

SCALES = [1.0, 1.4, 1.8]

# ==========================================
# PICAMERA2 SETUP
# ==========================================

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

# ==========================================
# AUTOFOCUS
# ==========================================

try:

    picam2.set_controls({
        "AfMode": 2
    })

except:

    print("Autofocus not supported")

time.sleep(2)

# ==========================================
# TRACKING VARIABLES
# ==========================================

smooth_x = FRAME_WIDTH // 2
smooth_y = FRAME_HEIGHT // 2

zoom_level = 1.0

locked_qr = None

last_seen_time = time.time()

prev_time = time.time()

# ==========================================
# MAIN LOOP
# ==========================================

while True:

    loop_start = time.time()

    # ==========================================
    # CAPTURE FRAME
    # ==========================================

    frame = picam2.capture_array()

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGB2BGR
    )

    frame = cv2.resize(
        frame,
        (
            FRAME_WIDTH,
            FRAME_HEIGHT
        )
    )

    original_frame = frame.copy()

    # ==========================================
    # FPS
    # ==========================================

    current_time = time.time()

    fps = 1 / max(
        current_time - prev_time,
        0.001
    )

    prev_time = current_time

    # ==========================================
    # PREPROCESSING
    # ==========================================

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # CLAHE CONTRAST
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # LIGHT SHARPEN
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    gray = cv2.filter2D(
        gray,
        -1,
        sharpen_kernel
    )

    # LIGHT DENOISE
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # ADAPTIVE THRESHOLD
    processed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # ==========================================
    # QR DETECTION
    # ==========================================

    qr_codes = []

    detection_scale = 1.0

    for scale in SCALES:

        resized = cv2.resize(
            processed,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        detected = decode(resized)

        if detected:

            qr_codes = detected
            detection_scale = scale

            break

    found_target = False

    # ==========================================
    # QR PROCESSING
    # ==========================================

    for qr in qr_codes:

        qr_data = qr.data.decode("utf-8")

        x, y, w, h = qr.rect

        # SCALE BACK
        x = int(x / detection_scale)
        y = int(y / detection_scale)
        w = int(w / detection_scale)
        h = int(h / detection_scale)

        # LOCK FIRST QR
        if locked_qr is None:

            locked_qr = qr_data

        # TRACK ONLY LOCKED QR
        if qr_data != locked_qr:

            continue

        found_target = True

        last_seen_time = time.time()

        # ==========================================
        # CENTER POSITION
        # ==========================================

        target_x = x + w // 2
        target_y = y + h // 2

        smooth_x += int(
            (target_x - smooth_x)
            * SMOOTHING
        )

        smooth_y += int(
            (target_y - smooth_y)
            * SMOOTHING
        )

        # ==========================================
        # AUTO ZOOM
        # ==========================================

        qr_size = max(w, h)

        if qr_size < 30:

            target_zoom = 3.5

        elif qr_size < 60:

            target_zoom = 2.8

        elif qr_size < 100:

            target_zoom = 2.0

        else:

            target_zoom = 1.2

        zoom_level += (
            target_zoom - zoom_level
        ) * ZOOM_SPEED

        zoom_level = max(
            MIN_ZOOM,
            min(MAX_ZOOM, zoom_level)
        )

        # ==========================================
        # CROP CALCULATION
        # ==========================================

        crop_w = int(
            FRAME_WIDTH / zoom_level
        )

        crop_h = int(
            FRAME_HEIGHT / zoom_level
        )

        center_x = smooth_x
        center_y = smooth_y

        # DEAD ZONE
        if abs(center_x - FRAME_WIDTH // 2) < DEAD_ZONE:

            center_x = FRAME_WIDTH // 2

        if abs(center_y - FRAME_HEIGHT // 2) < DEAD_ZONE:

            center_y = FRAME_HEIGHT // 2

        sx = max(
            center_x - crop_w // 2,
            0
        )

        sy = max(
            center_y - crop_h // 2,
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

        # SAFE CROP
        if ex <= sx or ey <= sy:

            continue

        cropped = original_frame[
            sy:ey,
            sx:ex
        ]

        # ==========================================
        # RESIZE ZOOM FRAME
        # ==========================================

        frame = cv2.resize(
            cropped,
            (
                FRAME_WIDTH,
                FRAME_HEIGHT
            ),
            interpolation=cv2.INTER_CUBIC
        )

        # ==========================================
        # BOX RECALCULATION
        # ==========================================

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

        # ==========================================
        # DISTANCE
        # ==========================================

        distance = (
            REAL_QR_WIDTH
            * FOCAL_LENGTH
        ) / max(w, 1)

        # ==========================================
        # ACCESS STATUS
        # ==========================================

        if qr_data == ALLOWED_QR:

            status = "ACCESS ALLOWED"
            color = (0, 255, 0)

        else:

            status = "ACCESS DENIED"
            color = (0, 0, 255)

        # ==========================================
        # DRAWING
        # ==========================================

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

        # ==========================================
        # DISPLAY TEXT
        # ==========================================

        cv2.putText(
            frame,
            "PROFESSIONAL QR TRACKING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            status,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Distance : {distance:.1f} cm",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Zoom : {zoom_level:.1f}x",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS : {int(fps)}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        break

    # ==========================================
    # SEARCH MODE
    # ==========================================

    if not found_target:

        cv2.putText(
            frame,
            "SEARCHING QR...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # SMOOTH RESET
        zoom_level += (
            1.0 - zoom_level
        ) * 0.05

        # RESET LOCK
        if time.time() - last_seen_time > 2:

            locked_qr = None

    # ==========================================
    # DISPLAY WINDOW
    # ==========================================

    cv2.imshow(
        "ADVANCED QR TRACKING",
        frame
    )

    # ==========================================
    # EXIT
    # ==========================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

    # ==========================================
    # FPS CONTROL
    # ==========================================

    elapsed = time.time() - loop_start

    delay = max(
        0,
        (1 / FPS_LIMIT) - elapsed
    )

    time.sleep(delay)

# ==========================================
# CLEANUP
# ==========================================

picam2.stop()

cv2.destroyAllWindows()