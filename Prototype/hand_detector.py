import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

detector = HandDetector(detectionCon=0.8, maxHands=1)
offset = 22
imgSize = 64  # Changed to match model's expected input size

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
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Add padding while keeping within frame bounds
        x_start = max(x - offset, 0)
        y_start = max(y - offset, 0)
        x_end = min(x + w + offset, img.shape[1])
        y_end = min(y + h + offset, img.shape[0])
        
        # Crop the hand region
        imgCrop = img[y_start:y_end, x_start:x_end]
        if imgCrop.size == 0:
            return [], [], img
            
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

    return hand_crops, bboxes, img

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
