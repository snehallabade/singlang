print("Starting app initialization...")

# Import required packages
try:
    print("Importing system packages...")
    import os
    import logging
    import traceback
    import atexit
    from time import time
    print("System packages imported successfully")

    print("Importing Flask...")
    from flask import Flask, render_template, Response, jsonify, request
    print("Flask imported successfully")

    print("Importing OpenCV and numpy...")
    import cv2
    import numpy as np
    print("OpenCV and numpy imported successfully")

    print("Importing mediapipe and cvzone...")
    import mediapipe
    from cvzone.HandTrackingModule import HandDetector
    print("Mediapipe and cvzone imported successfully")

    print("Importing local modules...")
    from hand_detector import detect_and_crop
    from sign_recognizer import predict_sign, clear_buffer
    from tts_engine import speak
    print("Local modules imported successfully")
except Exception as e:
    print(f"Error during imports: {str(e)}")
    raise

# Global variables
cap = None
detector = None
detected_letter = "No detection"  # Initialize with fallback text
last_detection_time = 0
DETECTION_COOLDOWN = 1.0  # seconds between detections

# Initialize Flask app
app = Flask(__name__, 
          template_folder='templates',
          static_folder='static')
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching during development

def initialize_camera():
    """Initialize the camera capture"""
    global cap
    try:
        if cap is not None:
            cap.release()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Video capture initialized")
        return True
    except Exception as e:
        print(f"Camera initialization error: {str(e)}")
        return False

def initialize_detector():
    """Initialize the hand detector"""
    global detector
    try:
        detector = HandDetector(detectionCon=0.8, maxHands=1)
        print("Hand detector initialized")
        return True
    except Exception as e:
        print(f"Detector initialization error: {str(e)}")
        return False

def process_frame(frame):
    """Process a single frame to detect hands and predict signs"""
    global detector, detected_letter, last_detection_time
    
    try:
        # Find hands in the frame
        hands, frame = detector.findHands(frame)
        
        if hands:
            # Process detected hands
            hand_crops, bboxes, processed_frame = detect_and_crop(frame)
            
            if hand_crops:
                # Take the first hand crop for prediction
                hand_crop = hand_crops[0]
                
                # Get prediction
                predicted_letter, confidence = predict_sign(hand_crop)
                current_time = time()
                
                # Only update detection if enough time has passed and we have a valid prediction
                if predicted_letter and current_time - last_detection_time >= DETECTION_COOLDOWN:
                    detected_letter = predicted_letter
                    last_detection_time = current_time
                
                # Draw prediction and confidence on frame
                bbox = bboxes[0]
                x, y = bbox[0], bbox[1]
                cv2.putText(frame, f"Prediction: {detected_letter}", (x-10, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x-10, y-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            # Clear buffer when no hand is detected
            clear_buffer()
        
        return frame
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        traceback.print_exc()
        return frame

def generate_frames():
    """Generate frames for the video feed"""
    while True:
        try:
            if cap is None or not cap.isOpened():
                print("Camera not available")
                initialize_camera()
                continue
                
            success, frame = cap.read()
            if not success or frame is None:
                print("Failed to read frame")
                continue
                
            # Process the frame
            processed_frame = process_frame(frame)
            
            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            continue

# Flask routes
@app.route('/')
def index():
    """Render the main page"""
    return render_template('model.html')

@app.route('/video_feed')
def video_feed():
    """Stream the video feed"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_letter')
def get_letter():
    """Get the currently detected letter"""
    global detected_letter
    return jsonify({'letter': detected_letter})

def cleanup():
    """Cleanup resources on shutdown"""
    global cap
    if cap is not None:
        try:
            print("Releasing camera...")
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released successfully")
        except Exception as e:
            print(f"Error releasing camera: {e}")

# Main entry point
if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.DEBUG)
    
    print("\nInitializing components...")
    try:
        if not initialize_detector():
            raise RuntimeError("Failed to initialize hand detector")
        if not initialize_camera():
            raise RuntimeError("Failed to initialize camera")
            
        # Register cleanup function
        atexit.register(cleanup)
        
        print("\nStarting Flask development server...")
        app.run(
            host='127.0.0.1',
            port=5000,
            debug=True,
            use_reloader=False  # Disable reloader to prevent camera issues
        )
    except Exception as e:
        print(f"Initialization error: {str(e)}")
        traceback.print_exc()
    finally:
        cleanup()
