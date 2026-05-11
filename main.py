from picamera2 import Picamera2, preview
from time import sleep

picam2 = Picamera2()
picam2.start_preview(preview.QTGL)
picam2.start()
sleep(5)
picam2.close()
