#!/usr/bin/env python3
import cv2
import numpy as np
import sys
from picamera2 import Picamera2

try:
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except ImportError:
    PICAM_AVAILABLE = False

def main():
    if not PICAM_AVAILABLE:
        print("Error: picamera2 not found. This script requires a Raspberry Pi camera.")
        return 1

    # Initialize HOG descriptor for person detection
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    picam2 = Picamera2()
    # Use a smaller resolution for better performance on Pi
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()

    print("Person detector started (using HOG). Press Ctrl+C to stop.")
    print("Point the camera at a person to see detections.")
    
    try:
        while True:
            frame_rgb = picam2.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # Detect people
            # winStride and padding can be tuned for performance/accuracy
            (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)

            for (x, y, w, h) in rects:
                print(f"Detected person at: x={x}, y={y}, w={w}, h={h}")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        picam2.stop()

if __name__ == "__main__":
    main()
