import cv2
from picamera2 import Picamera2
import sys

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888"}))
picam2.start()

detector = cv2.QRCodeDetector()

print("Camera started. Hold a QR code in front. Press Ctrl+C to exit.")

try:
    while True:
        frame = picam2.capture_array()
        data, bbox, _ = detector.detectAndDecode(frame)
        if data:
            print("QR Data:", data)
            break
except KeyboardInterrupt:
    pass
finally:
    picam2.close()
    cv2.destroyAllWindows()
