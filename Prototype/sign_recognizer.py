import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Enable Intel MKL-DNN optimization

from tensorflow.keras.models import load_model
import numpy as np
from collections import deque
import cv2

# Load model and labels once on import
model = load_model('cnn_bdlstm_model.h5', compile=False)
print(f"Model input shape: {model.input_shape}")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize frame buffers for both hands
left_frame_buffer = deque(maxlen=5)
right_frame_buffer = deque(maxlen=5)

def preprocess_image(img):
    """
    Preprocess image for better prediction.
    """
    # Ensure correct size
    if img.shape[:2] != (64, 64):
        img = cv2.resize(img, (64, 64))
    
    # Convert to float and normalize
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    
    # Enhance contrast
    img = (img - img.min()) / (img.max() - img.min())
    
    return img

def predict_sign(img, hand_type="Right"):
    """
    Predict the sign from the input image using a sequence of frames.
    Args:
        img: Input image of shape (64,64,3)
        hand_type: String indicating "Left" or "Right" hand
    Returns:
        tuple: (predicted_letter, confidence)
    """
    global left_frame_buffer, right_frame_buffer
    frame_buffer = left_frame_buffer if hand_type == "Left" else right_frame_buffer
    
    try:
        # Preprocess image
        img = preprocess_image(img)
        
        # Add to frame buffer
        frame_buffer.append(img)
        
        # If we don't have enough frames yet, return empty prediction
        if len(frame_buffer) < 5:
            return "", 0.0
        
        # Stack frames into sequence
        sequence = np.stack(list(frame_buffer))
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        
        # Make prediction
        try:
            prediction = model.predict(sequence, verbose=0)[0]
            predicted_index = np.argmax(prediction)
            confidence = float(prediction[predicted_index])
            predicted_letter = chr(predicted_index + ord('A'))
            
            # Get top 3 predictions
            top_3_indices = np.argsort(prediction)[-3:][::-1]
            top_3_predictions = [(chr(idx + ord('A')), float(prediction[idx])) 
                               for idx in top_3_indices]
            
            # Very high confidence - return immediately
            if confidence >= 0.85:
                return predicted_letter, confidence
                
            # Medium confidence - check if significantly better than second best
            if confidence >= 0.5 and len(top_3_predictions) > 1:
                confidence_gap = confidence - top_3_predictions[1][1]
                if confidence_gap > 0.3:  # Require larger gap between top predictions
                    return predicted_letter, confidence
                    
            # If same prediction persists with decent confidence
            if len(frame_buffer) == 5 and confidence >= 0.4:
                # Check if all frames in buffer predict the same letter
                all_predictions = model.predict(np.stack([f[np.newaxis, ...] for f in frame_buffer]), verbose=0)
                pred_letters = [chr(np.argmax(p) + ord('A')) for p in all_predictions]
                if all(l == predicted_letter for l in pred_letters):
                    return predicted_letter, confidence
            
            # Low confidence or not significantly better than second best
            return "", 0.0
            
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return "", 0.0
            
    except Exception as e:
        print(f"Error during frame processing: {e}")
        return "", 0.0

def clear_buffer(hand_type="Both"):
    """Clear the frame buffer(s) when hand tracking is lost"""
    global left_frame_buffer, right_frame_buffer
    if hand_type in ["Both", "Left"]:
        left_frame_buffer.clear()
    if hand_type in ["Both", "Right"]:
        right_frame_buffer.clear()

if __name__ == "__main__":
    # Placeholder for testing with sample image
    pass
