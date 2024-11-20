from ultralytics import YOLO
import cv2
import math
import cvzone
import os

# Load YOLO models
motorcycle_model = YOLO('yolov8m.pt')  # Model for detecting motorcycles
helmet_model = YOLO("Weights/best.pt")  # Model for detecting helmets
helmet_model.info()

# Define class names for helmet detection
helmet_classNames = ['Helmet', 'No Helmet']

def detect(source_path=None):
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
    orgImg = cv2.imread(image_path)
    orgImg = cv2.resize(orgImg, (0,0), fx=0.75, fy=0.75) 
    img = orgImg.copy()
    process_frame(img,orgImg)
    cv2.imshow("Detection", img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

def process_frame(img,orgImg):
    # Step 1: Detect motorcycles
    motorcycle_results = motorcycle_model(orgImg)
    motorcycles_detected = False

    for result in motorcycle_results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # Get class ID
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Check if the detected object is a motorcycle
            if cls == 3 and conf > 0.3:  # MOTORCYCLE_CLASS_ID  = 3 in yolov8 model
                motorcycles_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=20, t=2)
                cvzone.putTextRect(img, f'Motorcycle {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(0, 255, 0))

    # Step 2: If motorcycles are detected, detect helmets
    if motorcycles_detected:
        
        helmet_results = helmet_model(orgImg)
        for result in helmet_results:
            boxes = result.boxes
            # print(boxes)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if conf > 0.1:
                    cvzone.cornerRect(img, (x1, y1, w, h), l=20, t=2)
                    cvzone.putTextRect(img, f'{helmet_classNames[cls]} {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(255, 0, 0), colorT=(0, 0, 0))
    else:
        print("Motorcycle Not found.")

# Run detection
detect("Media/Test8.jpeg")
