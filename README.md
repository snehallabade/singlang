# Indian Sign Language Translation System

This project implements a real-time Indian Sign Language (ISL) translation system using computer vision and deep learning. The system can recognize hand gestures corresponding to English alphabets, Hindi vowels, and numerals.

## Features

- Real-time hand detection and tracking
- Sign language recognition using CNN-BDLSTM model
- Support for:
  - English Alphabets (A-Z)
  - Hindi Vowels
  - Numerals (0-9)
- Web interface for real-time interaction
- Text-to-speech output

## Project Structure

```
demo/
├── Prototype/
│   ├── app.py               # Main Flask application
│   ├── hand_detector.py     # Hand detection module
│   ├── sign_recognizer.py   # Sign recognition module
│   ├── tts_engine.py        # Text-to-speech module
│   ├── train_cnn_bdlstm.py  # Model training script
│   ├── static/             # Static files for web interface
│   └── templates/          # HTML templates
└── data/                   # Training and testing data
```

## Requirements

- Python 3.11
- TensorFlow 2.20.0
- OpenCV 4.12.0.88
- Flask 3.1.2
- cvzone 1.6.1
- MediaPipe
- pyttsx3

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kamal-stark-dev/Indian-Sign-Language-Translation-SIH.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python Prototype/app.py
   ```

## Model Architecture

The system uses a CNN-BDLSTM (Convolutional Neural Network with Bidirectional Long Short-Term Memory) architecture for sign recognition. The model processes sequences of frames to make predictions, which helps in capturing the temporal aspects of signs.

## Contributing

Feel free to open issues or submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
