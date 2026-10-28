# Face Emotion Detection Using CNN

A real-time facial emotion detection project built using Python, TensorFlow, Keras, and OpenCV.

## Overview

This project uses a Convolutional Neural Network (CNN) to predict facial expression categories from images captured through a webcam.

### Supported Emotion Classes

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- CNN (Convolutional Neural Network)

## Project Files

| File | Description |
|---|---|
| `train_model.py` | Trains the CNN model |
| `detect_emotion.py` | Runs real-time emotion detection |
| `check_labels.py` | Checks emotion label information |
| `emotion_model.h5` | Saved trained model |

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/Brindhasellakannu/Face-Emotion-Detection.git
cd Face-Emotion-Detection
```

### 2. Install dependencies

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

### 3. Run the webcam detection script

Make sure your webcam is connected, then run:

```bash
python detect_emotion.py
```

## Limitations

- Predictions may be affected by lighting, camera quality, and facial positioning.
- The model may misclassify some expressions.
- Predicted facial expressions do not necessarily reflect a person's actual emotional state.

## Future Improvements

- Improve model accuracy.
- Add real-time visual filters and facial overlays.
- Improve performance under different lighting conditions.

## Author

**Brindha Sellakkannu**

[GitHub Profile](https://github.com/Brindhasellakannu)
