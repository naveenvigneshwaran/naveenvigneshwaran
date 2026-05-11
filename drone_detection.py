import signal
import sys
import argparse
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

def main():
    global running
    
    parser = argparse.ArgumentParser(description='Drone Object Detection')
    parser.add_argument('--source', type=str, default='picamera',
                        choices=['picamera', 'usb', 'video'],
                        help='Camera source: picamera (default), usb, or video file')
    parser.add_argument('--device', type=int, default=0,
                        help='USB camera device ID (default: 0)')
    parser.add_argument('--video', type=str,
                        help='Path to video file (if source=video)')
    parser.add_argument('--width', type=int, default=640,
                        help='Frame width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Frame height (default: 480)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display (for headless/drone deployment)')
    parser.add_argument('--output', type=str,
                        help='Output video file path (optional)')
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt",
                                   "mobilenet_iter_73000.caffemodel")

    # Setup camera source
    cap = None
    picam2 = None
    
    if args.source == 'picamera':
        print(f"Using Pi Camera ({args.width}x{args.height})...")
        picam2 = Picamera2()
        picam2.configure(picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (args.width, args.height)}))
        picam2.start()
    elif args.source == 'usb':
        print(f"Using USB Camera (device {args.device})...")
        cap = cv2.VideoCapture(args.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print("Error: Cannot open USB camera")
            sys.exit(1)
    elif args.source == 'video':
        if not args.video:
            print("Error: Video file path required")
            sys.exit(1)
        print(f"Using video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("Error: Cannot open video file")
            sys.exit(1)

    print("Starting detection... Press 'q' to quit, or Ctrl+C to exit")
    print("Detection running. Press 'q' in the video window to quit.")
    print("If no display available, detection will run for 10 seconds and show stats...")

    try:
        frame_count = 0
        while running:
            if args.source == 'picamera':
                frame = picam2.capture_array()
            else:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or camera error")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

            # Convert back to BGR for OpenCV display
            if args.source == 'picamera':
                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                display_frame = frame

            cv2.imshow("Drone Object Detection", display_frame)
            frame_count += 1

            # Check for exit - only check window property after a few frames
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("'q' pressed, exiting...")
                break
            # Only check window property after 10 frames to allow window to initialize
            if frame_count > 10 and cv2.getWindowProperty("Drone Object Detection", 0) < 0:
                print("Window closed, exiting...")
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        running = False
        cv2.destroyAllWindows()
        if picam2:
            picam2.stop()
        if cap:
            cap.release()
        print("Cleanup done. Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
