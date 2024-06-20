import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
import easyocr
import csv
import os

cap = cv2.VideoCapture("2103099-uhd_3840_2160_30fps.mp4")

# Load the YOLO model
model = YOLO('best (9).pt')
classnames = ['license-plate', 'vehicle']


reader = easyocr.Reader(['en'], gpu=False)  # Specify language(s) and GPU usage


output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


output_csv_path = os.path.join(output_dir, 'detected_license_plates.csv')
csv_file = open(output_csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Frame', 'License Plate Text'])


plates_dir = os.path.join(output_dir, 'license_plate_images')
os.makedirs(plates_dir, exist_ok=True)

frame_number = 0


def preprocess_license_plate(plate_img):
    # Convert to grayscale
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    aspect_ratio = width / height
    new_height = 64
    new_width = int(aspect_ratio * new_height)
    resized = cv2.resize(gray, (new_width, new_height))


    thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


    denoised = cv2.bilateralFilter(thresh, 9, 75, 75)


    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)


    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(morphed, -1, sharpen_kernel)

    return sharpened


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # Resize the frame for consistent processing
    frame = cv2.resize(frame, (1080, 720))

    # Perform object detection
    results = model(frame)

    highest_confidence = 0
    best_plate_img = None
    best_text = ""
    best_bbox = None

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Only process if the detected class is 'license-plate' and confidence is above threshold
            if conf > 50 and class_detect == 'license-plate' and conf > highest_confidence:
                highest_confidence = conf
                # Extract the license plate region for OCR
                license_plate_img = frame[y1:y2, x1:x2]

                # Preprocess the license plate image
                preprocessed_plate_img = preprocess_license_plate(license_plate_img)

                # Perform OCR on the preprocessed license plate region
                ocr_results = reader.readtext(preprocessed_plate_img, detail=0, paragraph=True)

                if ocr_results:
                    best_text = ocr_results[0]
                    best_plate_img = license_plate_img
                    best_bbox = (x1, y1, x2, y2)

    if best_plate_img is not None:
        x1, y1, x2, y2 = best_bbox
        # Draw bounding box with custom color
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color
        # Draw custom text on the bounding box
        cvzone.putTextRect(frame, 'Detected Plate', [x1 + 8, y1 - 12], thickness=2, scale=1, colorR=(0, 0, 255),
                           colorB=(255, 255, 255))  # Red text on white background
        # Draw the recognized text on the frame with custom color
        cvzone.putTextRect(frame, best_text, [x1, y2 + 20], thickness=2, scale=1, colorR=(0, 255, 0),
                           colorB=(255, 255, 255))  # Green text on white background
        # Save the result to the CSV file
        csv_writer.writerow([frame_number, best_text])
        # Optionally save the license plate image
        plate_image_path = os.path.join(plates_dir, f'frame_{frame_number}_plate.png')
        cv2.imwrite(plate_image_path, best_plate_img)

    # Display the frame with annotations
    cv2.imshow('frame', frame)

    # Exit loop on pressing 't'
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# Release video capture object and close all windows
cap.release()
csv_file.close()
cv2.destroyAllWindows()