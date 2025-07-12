# Budget-Manager-App
Track your expenses, save smarter, and take control of your finances

## Abstract
The project focuses on recognizing human facial expressions to improve human-computer interaction (HCI). It uses facial images collected from Unsplash, detects facial regions using OpenCV and DeepFace, and classifies expressions using a custom-built CNN model with six layers. The model was trained and tested to detect seven universal emotions: Happy, Sad, Angry, Fear, Surprise, Disgust, and Neutral.

## Problem Statement
While humans are naturally adept at reading emotions, automating this with machines is a challenge. This system aims to mimic that capability by:
1. Detecting faces in images
2. Extracting facial features
3. Analyzing and classifying expressions into emotional categories
This has applications in retail, healthcare, entertainment, and user experience design.

## Scope of the Project
Facial expression recognition can be used in:
- Virtual reality, online surveys, webinars
- Security, surveillance, forensic systems
- Human-computer interaction
- Healthcare for monitoring emotional states

Traditional methods struggle with delay and poor accuracy. This project uses CNN and DeepFace for improved efficiency and performance in real-time applications.

## Tools & Technologies
**CNN Architecture**
- 6 Layers Total:
   - 4 Convolutional layers (with Batch Normalization, ReLU, MaxPooling, Dropout)
   - 2 Fully connected (Dense) layers + SoftMax Output
- Flatten Layer: Converts image features into a vector
- Adam Optimizer: For fast convergence
- Loss Function: Categorical Cross-Entropy
- Metric: Accuracy

**Hardware & Software**
- Hardware: Intel/Ryzen processor, 8GB RAM
- Software: Python, OpenCV, Google Colab, Jupyter Notebook
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, Keras

## Dataset & Training
- Dataset: FER-2013 (35,887 grayscale images of size 48x48)
- Split: 28,709 training + 7,178 testing images
- Training Method:
   - Mini-Batch Gradient Descent
   - Data Augmentation via ImageDataGenerator
   - Trained in Google Colab using Python and Keras

## System Implementation
- Face Detection: OpenCV used to identify and extract facial regions
- Real-Time Classification: Implemented via a Flask app for webcam-based emotion recognition
- Model Integration: Predictions are visualized on detected faces with bounding boxes

## Results
- Validation Accuracy: Peaked between 60% – 65%, with a final test accuracy of 62.7%
- Best Performance: On Happy and Surprised faces
- Poor Performance: On Fear (often confused with Sad)
- Confusion Matrix: Used for performance evaluation

## Conclusion & Future Enhancements
Achieved results are close to industry benchmark (65% ±5%) on FER-2013

**Future improvements:**
- Better illumination adaptation and noise filtering
- Hyperparameter tuning and early stopping
- Testing the model on other datasets
- Adding more CNN layers and increasing epochs with care to avoid overfitting
- Exploring other model architectures like ResNet or MobileNet

