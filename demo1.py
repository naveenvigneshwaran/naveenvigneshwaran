import cv2
import numpy as np
from picamera2 import Picamera2

# ---------------- CAMERA ----------------
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)

picam2.configure(config)
picam2.start()

cv2.namedWindow("Vending QR System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Vending QR System", 640, 480)

# ---------------- QR DETECTOR ----------------
detector = cv2.QRCodeDetector()

# ---------------- ONLY VALID VENDING QR ----------------
VALID_VENDING_QR = "QR.png"

while True:

    frame = picam2.capture_array()

    data, bbox, _ = detector.detectAndDecode(frame)

    # ---------------- NO QR ----------------
    if not data or bbox is None:
        cv2.putText(frame, "NO VALID QR", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    else:

        bbox = bbox[0].astype(int)

        # ---------------- CHECK QR ----------------
        if data == VALID_VENDING_QR:
            status = "VENDING ACCESS GRANTED"
            color = (0, 255, 0)

        else:
            status = "INVALID QR - REJECTED"
            color = (0, 0, 255)

        # ---------------- DRAW BOX ----------------
        for i in range(4):
            cv2.line(frame,
                     tuple(bbox[i]),
                     tuple(bbox[(i + 1) % 4]),
                     color, 3)

        # ---------------- TEXT ----------------
        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"QR DATA: {data}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # ---------------- ACTION SIMULATION ----------------
        if data == VALID_VENDING_QR:
            cv2.putText(frame, ">> DISPENSE PRODUCT", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Vending QR System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()