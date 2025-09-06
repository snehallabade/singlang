import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

detector = HandDetector(detectionCon=0.8, maxHands=2)  # Allow detection of both hands
offset = 22
imgSize = 64  # Changed to match model's expected input size
font = cv2.FONT_HERSHEY_SIMPLEX  # Font for drawing text

def detect_and_crop(frame):
    """
    Detect hands in the frame and crop them with improved preprocessing.
    Args:
        frame: Input frame from video capture
    Returns:
        tuple: (hand_crops, bboxes, processed_frame)
    """
    hands, img = detector.findHands(frame)
    hand_crops = []
    bboxes = []
    hand_types = []  # To track which hand is which (Left/Right)
    
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            hand_type = "Right" if hand["type"] == "Right" else "Left"
            hand_types.append(hand_type)
        
            # Add padding while keeping within frame bounds
            x_start = max(x - offset, 0)
            y_start = max(y - offset, 0)
            x_end = min(x + w + offset, img.shape[1])
            y_end = min(y + h + offset, img.shape[0])
            
            # Crop the hand region
            imgCrop = img[y_start:y_end, x_start:x_end]
            if imgCrop.size == 0:
                continue  # Skip this hand if crop is empty
            
            # Create white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w
            
            # Preserve aspect ratio while resizing
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                # Center the image horizontally
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
                
                # Apply mild contrast enhancement
                imgWhite = cv2.convertScaleAbs(imgWhite, alpha=1.1, beta=0)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                # Center the image vertically
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize
                
                # Apply mild contrast enhancement
                imgWhite = cv2.convertScaleAbs(imgWhite, alpha=1.1, beta=0)
            
            # Apply Gaussian blur to reduce noise
            imgWhite = cv2.GaussianBlur(imgWhite, (3,3), 0)
            
            hand_crops.append(imgWhite)
            bboxes.append(hand['bbox'])
            
            # Draw hand type label on the image
            cv2.putText(img, hand_type, (x, y - 10), font, 1, (255, 0, 0), 2)

    return hand_crops, bboxes, img, hand_types

if __name__ == "__main__":
    # Test code if needed
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        crops, boxes, img = detect_and_crop(frame)
        for crop in crops:
            cv2.imshow("Crop", crop)
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
