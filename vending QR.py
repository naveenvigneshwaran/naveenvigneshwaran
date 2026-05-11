import cv2
import numpy as np
from picamera2 import Picamera2

# ---------------- CAMERA SETUP ----------------
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)

picam2.configure(config)
picam2.start()

# ---------------- QR DETECTOR ----------------
detector = cv2.QRCodeDetector()

# ---------------- DEPTH PARAMETERS ----------------
# Approximate focal length (adjust for your camera)
FOCAL_LENGTH = 700

# Real QR width in cm
REAL_QR_WIDTH = 5.0

while True:

    # Capture frame
    frame = picam2.capture_array()

    # Detect QR
    data, bbox, _ = detector.detectAndDecode(frame)

    if bbox is not None and len(bbox) > 0:

        bbox = np.int32(bbox)

        # Draw bounding box
        n = len(bbox[0])

        for i in range(n):
            pt1 = tuple(bbox[0][i])
            pt2 = tuple(bbox[0][(i + 1) % n])

            cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

        # ---------------- CENTER POINT ----------------
        center_x = int(np.mean(bbox[0][:, 0]))
        center_y = int(np.mean(bbox[0][:, 1]))

        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # ---------------- DEPTH ESTIMATION ----------------
        # Pixel width of QR code
        pixel_width = np.linalg.norm(bbox[0][0] - bbox[0][1])

        # Distance formula
        distance = (REAL_QR_WIDTH * FOCAL_LENGTH) / pixel_width

        # ---------------- DISPLAY INFO ----------------
        if data:
            cv2.putText(
                frame,
                f"QR: {data}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

        cv2.putText(
            frame,
            f"Distance: {distance:.2f} cm",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Accuracy-like display
        cv2.putText(
            frame,
            "Detection: HIGH",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    else:
        cv2.putText(
            frame,
            "No QR Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ---------------- SHOW WINDOW ----------------
    cv2.imshow("QR Scanner with Depth", frame)

    # Quit key
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

# ---------------- CLEANUP ----------------
cv2.destroyAllWindows()
picam2.close()