from ultralytics import YOLO
import cv2
import math
import cvzone
import os
import time  # Import time module to calculate running times

# Load YOLO models
motorcycle_model = YOLO('yolov8m.pt')  # Model for detecting motorcycles
helmet_model = YOLO("Weights/best.pt")  # Model for detecting helmets
helmet_model.info()

# Define class names for helmet detection
helmet_classNames = ['Helmet', 'No Helmet']

# Directory to save cropped images
output_dir = "Cropped_Results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Global variables to store bounding boxes
motorcycle_bboxes = []  # List to store motorcycle bounding boxes
helmet_bboxes = []      # List to store helmet bounding boxes


def detect(image_path):
    orgImg = cv2.imread(image_path)
    orgImg = cv2.resize(orgImg, (0, 0), fx=0.75, fy=0.75)
    img = orgImg.copy()
    process_frame(img, orgImg)
    cv2.imshow("Detection", img)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()


def process_frame(img, orgImg):
    global motorcycle_bboxes, helmet_bboxes  # Access global lists

    # Clear the arrays for fresh detections
    motorcycle_bboxes.clear()
    helmet_bboxes.clear()

    # Measure total processing time
    total_start_time = time.time()

    # Step 1: Detect motorcycles
    motorcycle_start_time = time.time()
    motorcycle_results = motorcycle_model(orgImg)
    motorcycle_end_time = time.time()
    motorcycle_time = motorcycle_end_time - motorcycle_start_time  # Time to detect motorcycles

    motorcycles_detected = False

    for result in motorcycle_results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])  # Get class ID
            conf = math.ceil((box.conf[0] * 100)) / 100

            # Check if the detected object is a motorcycle
            if cls == 3 and conf > 0.3:  # MOTORCYCLE_CLASS_ID = 3 in YOLOv8 model
                motorcycles_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                motorcycle_bboxes.append([x1, y1, x2, y2])  # Save bounding box coordinates
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=20, t=2)
                cvzone.putTextRect(img, f'Motorcycle {conf}', (x1, y1 - 10), scale=0.8, thickness=1, colorR=(0, 255, 0))

    # Print motorcycle bounding boxes
    if motorcycles_detected:
        print("Motorcycle Bounding Boxes:")
        for bbox in motorcycle_bboxes:
            print(bbox)

    # Step 2: If motorcycles are detected, detect helmets
    helmet_time = 0

    if motorcycles_detected:
        helmet_start_time = time.time()
        helmet_results = helmet_model(orgImg)

        # Step 1: Detect and save all helmet bounding boxes
        for result in helmet_results:
            boxes = result.boxes
            for box in boxes:
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])  # Class ID (Helmet or No Helmet)

                if conf > 0.1:  # Confidence threshold
                    # Save helmet bounding box coordinates
                    hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                    helmet_bboxes.append([hx1, hy1, hx2, hy2])

        # Print all detected helmet bounding boxes
        print("Detected Helmet Bounding Boxes (Unfiltered):")
        for bbox in helmet_bboxes:
            print(bbox)

        # Step 2: Match motorcycle coordinates with helmet coordinates
        filtered_helmet_bboxes = []  # Store helmets that pass the motorcycle check

        for mx1, my1, mx2, my2 in motorcycle_bboxes:
            for hx1, hy1, hx2, hy2 in helmet_bboxes[:]:  # Iterate over a copy to modify the original list
                # Helmet's center
                helmet_center_x = (hx1 + hx2) / 2

                # Check if helmet's center is horizontally within the motorcycle
                # and if the helmet is reasonably above the motorcycle or overlaps slightly
                if mx1 < helmet_center_x < mx2 and hy2 <= my2 + 100:  # Adding tolerance
                    # Helmet matches the motorcycle
                    filtered_helmet_bboxes.append([hx1, hy1, hx2, hy2])

                    # Draw bounding box for the matched helmet
                    w, h = hx2 - hx1, hy2 - hy1
                    cvzone.cornerRect(img, (hx1, hy1, w, h), l=20, t=2)
                    cvzone.putTextRect(
                        img,
                        f'{helmet_classNames[cls]} {conf}',
                        (hx1, hy1 - 10),
                        scale=0.8,
                        thickness=1,
                        colorR=(255, 0, 0),
                        colorT=(0, 0, 0)
                    )

                    # Remove the matched helmet from the helmet_bboxes list
                    helmet_bboxes.remove([hx1, hy1, hx2, hy2])

        # Print filtered helmet bounding boxes
        print("Filtered Helmet Bounding Boxes (Matched with Motorcycles):")
        for bbox in filtered_helmet_bboxes:
            print(bbox)

        # Print remaining helmet bounding boxes
        print("Remaining Helmet Bounding Boxes (Not Matched):")
        for bbox in helmet_bboxes:
            print(bbox)

        helmet_end_time = time.time()
        helmet_time = helmet_end_time - helmet_start_time  # Time to detect helmets

    else:
        print("Motorcycle Not found.")

    # Measure total processing time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time  # Total processing time

    # Print times
    print(f"Motorcycle Detection Time: {motorcycle_time:.2f} seconds")
    print(f"Helmet Detection Time: {helmet_time:.2f} seconds")
    print(f"Total Processing Time: {total_time:.2f} seconds")


# Run detection on an image
detect("Media/5.jpg")

# Access global arrays after detection
print("Final Motorcycle Bounding Boxes Array:", motorcycle_bboxes)
print("Final Helmet Bounding Boxes Array:", helmet_bboxes)
