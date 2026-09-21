# AI Safety Equipment Detection System

A computer vision application that detects safety equipment in images using a custom-trained YOLO11 object detection model.

The project demonstrates an end-to-end machine learning workflow including dataset preparation, YOLO model training, evaluation, inference, and deployment through an interactive Streamlit application.

## 🚀 Live Demo

Try the deployed application here:

https://ai-safety-equipment-detection.streamlit.app

## 🧠 Project Overview

AI Safety Equipment Detection System is a computer vision application that uses a custom-trained YOLO11 model to detect workplace safety equipment in images.

The model currently detects:

- Hard hats
- Safety vests

Users can upload an image through a Streamlit web interface, run real-time object detection, view bounding boxes and confidence scores, and download the detection result.

## 🛠️ Technologies

- Python
- Ultralytics YOLO11
- PyTorch
- OpenCV
- Streamlit
- Pillow
- Computer Vision
- Object Detection
- Custom Model Training

## ✨ Features

- Custom-trained YOLO11 object detection model
- Hard-hat detection
- Safety-vest detection
- Image upload and real-time inference
- Adjustable confidence threshold
- Bounding-box visualization
- Confidence scores
- Detection counts by class
- Downloadable detection results
- Streamlit web interface
- Cloud deployment

## 🤖 Machine Learning Workflow

1. Prepared and organized the object-detection dataset
2. Trained a YOLO11 model on labeled safety-equipment images
3. Evaluated model performance
4. Saved the best-performing model weights
5. Integrated the trained model into a Python application
6. Built an interactive Streamlit interface
7. Deployed the trained model and application to the cloud

## 📁 Model

The deployed application uses the custom-trained model:

`models/best.pt`

This allows the application to perform inference using the model produced during training rather than relying solely on a pretrained general-purpose model.

## Features

- Custom-trained YOLO11 object detection model
- Hard-hat detection
- Safety-vest detection
- Image upload and real-time inference
- Adjustable confidence threshold
- Bounding-box visualization
- Detection counts by class
- Downloadable detection results
- Image, video, and tracking utilities
- Streamlit web interface

## Model Performance

The final model was trained for 10 epochs and evaluated on a separate test set.

| Metric    | Overall |
| --------- | ------: |
| Precision |   0.875 |
| Recall    |   0.782 |
| mAP@50    |   0.867 |
| mAP@50-95 |   0.637 |

### Hard Hat

- Precision: 0.962
- Recall: 0.859
- mAP@50: 0.956
- mAP@50-95: 0.640

### Safety Vest

- Precision: 0.788
- Recall: 0.704
- mAP@50: 0.778
- mAP@50-95: 0.633

## Dataset

The model was trained using a hard-hat and safety-vest computer vision dataset prepared in YOLO format.

Prepared dataset:

- Training images: 5,127
- Validation images: 1,410
- Test images: 712
- Classes: `hard-hat`, `safety-vest`

The original dataset contained inconsistent annotations, so a preprocessing script was used to clean the labels and prepare a two-class object detection dataset.

## Technologies

- Python
- Ultralytics YOLO11
- PyTorch
- Streamlit
- Pillow
- Computer Vision
- Object Detection

## Project Structure

    ai-object-detection/
    ├── app.py
    ├── detect.py
    ├── video_detector.py
    ├── tracker.py
    ├── prepare_dataset.py
    ├── requirements.txt
    └── README.md

Large datasets, model weights, virtual environments, and generated training output are excluded from the GitHub repository.

## Run Locally

Create and activate a Python virtual environment, then install the dependencies:

    pip install -r requirements.txt

Run the Streamlit application:

    streamlit run app.py

Open the local Streamlit URL displayed in the terminal.

## Training

The custom YOLO11 model was trained using Ultralytics YOLO.

Example:

    yolo detect train model=yolo11n.pt data=dataset/data.yaml epochs=10 imgsz=416 batch=4

## Application

Upload an image through the Streamlit interface. The trained model performs inference and displays:

- Original image
- Annotated detection image
- Detected equipment classes
- Number of detections
- Confidence-based predictions

Users can adjust the confidence threshold and download the annotated detection result.

## Purpose

This project was created to demonstrate practical experience with:

- Computer vision
- Object detection
- Model training and evaluation
- Dataset preprocessing
- Machine learning inference
- Python application development
- Interactive ML deployment

## Author

Kevin Jordan

Full Stack Software Developer

GitHub: KevinJ3259
