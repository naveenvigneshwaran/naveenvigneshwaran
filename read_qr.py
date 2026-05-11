import cv2
import sys
import os
import numpy as np

path = os.path.join(os.path.dirname(__file__), "QR.png")

# Check if it's actually an SVG
with open(path, 'rb') as f:
    header = f.read(10)

if b'<svg' in header.lower() or b'<?xml' in header.lower():
    import cairosvg
    img_bytes = cairosvg.svg2png(url=path, output_width=500, output_height=500)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
else:
    img = cv2.imread(path)

if img is None:
    print("Could not read image")
    sys.exit(1)

detector = cv2.QRCodeDetector()
data, bbox, _ = detector.detectAndDecode(img)

if data:
    print("QR Data:", data)
else:
    print("No QR code found")
