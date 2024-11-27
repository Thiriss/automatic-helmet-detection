import cv2
import math
import cvzone
from ultralytics import YOLO
import os
import time

# Load YOLO model with custom weights
model = YOLO("Weights/bikeHelmet.pt")
model.info()

# Define class names
classNames = ['helmet', 'motorbike' , 'non-helmet']


def detect_helmets(source_path=None):
    if source_path is None:
        # Default to webcam if no file path is provided
        process_video(0)
    else:
        # Automatically detect the type of file
        ext = os.path.splitext(source_path)[-1].lower()

        # Common video file extensions
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        # Common image file extensions
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        if ext in video_exts:
            process_video(source_path)
        elif ext in image_exts:
            process_image(source_path)
        else:
            print("Unsupported file format. Please provide a valid image or video file.")


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        success, img = cap.read()
        if not success:
            break
        process_frame(img)
        cv2.imshow("Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def process_image(image_path):
    img = cv2.imread(image_path)
    process_frame(img)
    cv2.imshow("Detection", img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def process_frame(img):
    total_start_time = time.time()
    # Perform object detection
    results = model(img)
    # Loop through the detections and draw bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1
            # Adjust thickness of bounding box
            cvzone.cornerRect(img, (x1, y1, w, h), l=20, t=2)  # 't' is thickness, 'l' is length of corners

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if conf > 0.3:
                # Adjust text color, thickness, and box color
                color_map = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 255, 0)}  # Colors for each class
                color = color_map.get(cls, (255, 0, 0))  # Default to blue if class not found
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1,
                                   colorR=color, colorT=(0, 0, 0))
                
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total Processing Time: {total_time:.2f} seconds")

# detect file type and perform detection
detect_helmets("Media/dt3.jpg")