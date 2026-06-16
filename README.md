# Car Color Detection & Traffic Monitoring System

## Problem Statement

Traffic monitoring systems play an important role in intelligent transportation and smart city applications. The objective of this project is to develop a machine learning-based system capable of:

* Detecting vehicles in traffic scenes.
* Identifying the color of detected vehicles.
* Counting the number of vehicles present.
* Counting the number of pedestrians present.
* Highlighting blue-colored vehicles with red bounding boxes.
* Highlighting all other vehicles with blue bounding boxes.
* Providing an interactive graphical user interface (GUI) for image upload and visualization.

This project combines object detection and image classification techniques to create a complete traffic analysis solution.

---

## Dataset

### 1. Vehicle Color Recognition Dataset (VCoR)

The VCoR dataset was used to train the vehicle color classification model.

#### Color Classes

* Beige
* Black
* Blue
* Brown
* Gold
* Green
* Grey
* Orange
* Pink
* Purple
* Red
* Silver
* Tan
* White
* Yellow

The dataset was divided into:

* Training Set
* Validation Set
* Test Set

### 2. Intersection Traffic Dataset

The Intersection Traffic Dataset was used for testing the complete traffic monitoring pipeline. It contains traffic intersection images with vehicles, pedestrians, and traffic infrastructure.

---

## Methodology

### Step 1: Data Preprocessing

* Images were resized to 224 × 224 pixels.
* Pixel values were normalized to the range [0,1].
* TensorFlow ImageDataGenerator was used for efficient data loading.

### Step 2: Vehicle Color Classification

A MobileNetV2-based transfer learning model was implemented.

#### Model Architecture

* Base Model: MobileNetV2 (Pre-trained on ImageNet)
* GlobalAveragePooling2D Layer
* Dense Output Layer with Softmax Activation

#### Why MobileNetV2?

* Lightweight architecture
* Fast training and inference
* High performance with transfer learning
* Suitable for real-time applications

### Step 3: Vehicle and Pedestrian Detection

YOLOv8 was used for object detection.

#### Objects Detected

* Cars
* Pedestrians
* Traffic-related objects

#### Why YOLOv8?

* State-of-the-art object detection model
* High detection speed
* Real-time performance
* Accurate bounding box predictions

### Step 4: Integration

The complete pipeline works as follows:

Traffic Image
↓
YOLOv8 Detection
↓
Vehicle Cropping
↓
Color Classification using MobileNetV2
↓
Apply Custom Bounding Box Logic
↓
Count Vehicles and Pedestrians
↓
Display Results

### Custom Bounding Box Logic

* Blue Vehicle → Red Bounding Box
* Other Vehicle Colors → Blue Bounding Box

### Step 5: GUI Development

A Gradio-based GUI was developed to:

* Upload traffic images
* Preview input images
* Display processed images
* Visualize detected vehicles and pedestrians
* Show color predictions and counts

---

## Results

### Vehicle Color Classification

Model Used:

* MobileNetV2 Transfer Learning

Test Accuracy:

* 64.2%

### Traffic Monitoring Output

The system successfully:

* Detects vehicles in traffic scenes.
* Detects pedestrians.
* Predicts vehicle colors.
* Counts vehicles.
* Counts pedestrians.
* Highlights blue vehicles with red bounding boxes.
* Highlights other vehicles with blue bounding boxes.

### Sample Output Features

* Vehicle Color Labels
* Vehicle Count
* Pedestrian Count
* Color-Based Bounding Boxes
* Interactive GUI Output

---

## Technologies Used

* Python 3.11
* TensorFlow
* Keras
* MobileNetV2
* YOLOv8
* OpenCV
* NumPy
* Matplotlib
* Gradio
* Jupyter Notebook

---

## Project Structure

```text
car-color-detection-traffic-monitoring/
│
├── notebook.ipynb
├── car_color_model.h5
├── requirements.txt
├── README.md
├── screenshots/
│   ├── output1.png
│   ├── output2.png
│   └── gui.png
└── datasets/
```

---

## Future Improvements

* Improve color classification accuracy through fine-tuning.
* Support real-time video processing.
* Add vehicle tracking across frames.
* Deploy the application as a web service.
* Integrate traffic analytics and reporting features.

---

## Conclusion

This project demonstrates the integration of deep learning-based color classification and object detection techniques for traffic monitoring applications. By combining MobileNetV2 and YOLOv8, the system successfully detects vehicles and pedestrians, identifies vehicle colors, applies custom color-based visualization rules, and provides an interactive GUI for user-friendly operation.
