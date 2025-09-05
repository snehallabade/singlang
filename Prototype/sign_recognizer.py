import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Enable Intel MKL-DNN optimization

from tensorflow.keras.models import load_model
import numpy as np
from collections import deque

# Load model and labels once on import
model = load_model('cnn_bdlstm_model.h5', compile=False)
print(f"Model input shape: {model.input_shape}")
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize frame buffer
frame_buffer = deque(maxlen=5)

def predict_sign(img):
    """
    Predict the sign from the input image using a sequence of frames.
    Args:
        img: Input image of shape (64,64,3)
    Returns:
        tuple: (predicted_letter, confidence)
    """
    global frame_buffer
    
    try:
        # If image is already normalized (values between 0-1), use as is
        if img.max() > 1.0:
            img = img / 255.0
        
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
            if confidence >= 0.7:
                return predicted_letter, confidence
                
            # Medium confidence - check if significantly better than second best
            if confidence >= 0.3 and len(top_3_predictions) > 1:
                confidence_gap = confidence - top_3_predictions[1][1]
                if confidence_gap > 0.2:
                    return predicted_letter, confidence
            
            # Low confidence or not significantly better than second best
            return "", 0.0
            
        except Exception as e:
            print(f"Error during model prediction: {e}")
            return "", 0.0
            
    except Exception as e:
        print(f"Error during frame processing: {e}")
        return "", 0.0

def clear_buffer():
    """Clear the frame buffer when hand tracking is lost"""
    global frame_buffer
    frame_buffer.clear()

if __name__ == "__main__":
    # Placeholder for testing with sample image
    pass
