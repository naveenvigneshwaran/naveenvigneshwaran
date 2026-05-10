# OpenCV Object Detection Demo

This project runs object detection on files and face detection on the webcam
or files with OpenCV and saves the annotated result plus structured detections.

## Run

```bash
source .venv/bin/activate
python demo_opencv.py --input path/to/image.jpg
```

Use `demo_opencv.py` for image or video files. You can also pass an image URL:

```bash
python demo_opencv.py --input https://example.com/image.jpg
```

For a video file:

```bash
python demo_opencv.py --input path/to/video.mp4
```

For a webcam explicitly:

```bash
python demo_opencv.py --input 0 --display
```

For face detection on the webcam, use the dedicated entry point:

```bash
python face_opencv.py
```

You can also pass files to the face script:

```bash
python face_opencv.py --input path/to/image.jpg
```

### QR Code Detection

For QR code detection, use `qr_code.py`. It supports image files, standard USB cameras, and the Raspberry Pi camera module.

```bash
# Detect from an image
python qr_code.py --image path/to/image.jpg

# Detect from camera 0 (default if no source specified)
python qr_code.py

# Detect from Pi Camera Module (requires picamera2)
python qr_code.py --picamera

# Run in headless mode (no preview window)
python qr_code.py --headless --output detections.txt
```

To choose the export format:

```bash
python demo_opencv.py --input path/to/video.mp4 --export-format csv
python demo_opencv.py --input path/to/image.jpg --export-format both
```

## Output

- `outputs/annotated.jpg`
- `outputs/annotated.mp4`
- `outputs/detections.json`
- `outputs/detections.csv`

`demo_opencv.py` downloads the object-detection model files into `models/` on
first run. Face mode uses the built-in OpenCV Haar cascade.
