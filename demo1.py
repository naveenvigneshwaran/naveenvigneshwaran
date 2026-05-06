import signal
import sys
from picamera2 import Picamera2
import cv2
import numpy as np

# Global flag
running = True

def signal_handler(sig, frame):
    global running
    print("\nInterrupted! Stopping...")
    running = False

signal.signal(signal.SIGINT, signal_handler)

# Load class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Load model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                               "mobilenet_iter_73000.caffemodel")

# Camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

print("Starting detection... Press 'q' to quit, or Ctrl+C to exit")

try:
    while running:
        frame = picam2.capture_array()

        # Prepare image for detection
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                      0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        h, w = frame.shape[:2]

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

        cv2.imshow("Object Detection", frame)

        # Check for exit conditions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("'q' pressed, exiting...")
            break
        if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed, exiting...")
            break
except Exception as e:
    print(f"Error: {e}")
finally:
    running = False
    cv2.destroyAllWindows()
    picam2.stop()
    print("Cleanup done. Exiting.")
    sys.exit(0)