import cv2
import math
import cvzone
from ultralytics import YOLO

# Load YOLO model with custom weights
model = YOLO("Weights/best.pt")

# Define class names
classNames = ['With Helmet', 'Without Helmet']

def detect_helmets(source_type, source_path=None):
    if source_type == 'video':
        cap = cv2.VideoCapture(source_path or 0)  # If no path is provided, default to webcam
        while True:
            success, img = cap.read()
            if not success:
                break
            process_frame(img)
            cv2.imshow("Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()

    elif source_type == 'image':
        img = cv2.imread(source_path)
        process_frame(img)
        cv2.imshow("Detection", img)
        cv2.waitKey(0)  # Wait for a key press to close the window

    cv2.destroyAllWindows()

def process_frame(img):
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

            if conf > 0.1:
                # Adjust text color, thickness, and box color
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0), colorT=(0, 0, 0))

# Example usage:
# For video: provide source type 'video' and video path or 0 for webcam
detect_helmets('video', "Media/bike_rw_3.mp4")

# For image: provide source type 'image' and image path
# detect_helmets('image', "Media/image.png")
