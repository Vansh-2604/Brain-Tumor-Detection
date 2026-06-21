# 🧠 Brain Tumor Detection Using Deep Learning

## Overview

Brain Tumor Detection is an AI-powered web application designed to assist in the classification of brain MRI scans using Deep Learning. The system leverages Transfer Learning with EfficientNet-B0 to identify and classify brain tumors into multiple categories with high confidence.

The application provides an intuitive web interface where users can upload MRI images, receive real-time predictions, and view confidence scores through an interactive dashboard.

---

## Features

* 📤 MRI Image Upload and Preview
* 🧠 Automated Brain Tumor Classification
* 📊 Confidence Score Visualization
* 🎯 Real-Time Prediction
* 💻 Responsive and User-Friendly Interface
* 🔄 New Analysis Functionality
* 📷 MRI Scan Visualization
* ⚡ Fast Inference using PyTorch

---

## Tumor Categories

The model is capable of classifying MRI scans into the following categories:

* Glioma Tumor
* Meningioma Tumor
* Pituitary Tumor
* No Tumor Detected

---

## Technology Stack

### Machine Learning & Deep Learning

* Python
* PyTorch
* Torchvision
* EfficientNet-B0
* Transfer Learning

### Web Development

* Flask
* HTML5
* CSS3
* Bootstrap 5

### Data Processing & Visualization

* NumPy
* Matplotlib
* Seaborn
* Pillow (PIL)

---

## Model Architecture

The project utilizes **EfficientNet-B0**, a state-of-the-art convolutional neural network architecture known for its excellent balance between accuracy and computational efficiency.

### Workflow

MRI Scan → Image Preprocessing → EfficientNet-B0 Model → Prediction → Confidence Score → Result Dashboard

---

## Project Structure

```text
Brain-Tumor-Detection/
│
├── app.py
├── predict.py
├── requirements.txt
├── README.md
│
├── model/
│   └── brain_tumor_model.pth
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
│
├── templates/
│   └── index.html
│
└── .gitignore
```

## Installation

### Clone the Repository

```bash
git clone https://github.com/Vansh-2604/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## SCREENSHOT

![image alt](https://github.com/Vansh-2604/Brain-Tumor-Detection/blob/3ddd8b8987bba1004f0525fb6fecd7826d301534/ss1.jpg)
![image alt](https://github.com/Vansh-2604/Brain-Tumor-Detection/blob/3ddd8b8987bba1004f0525fb6fecd7826d301534/ss2.jpg)
![image alt](https://github.com/Vansh-2604/Brain-Tumor-Detection/blob/3ddd8b8987bba1004f0525fb6fecd7826d301534/ss3.jpg)
![image alt](https://github.com/Vansh-2604/Brain-Tumor-Detection/blob/3ddd8b8987bba1004f0525fb6fecd7826d301534/ss4.jpg)



## Author

**Vansh Sharma**

B.Tech Computer Science Engineering

Passionate about Artificial Intelligence, Machine Learning, Deep Learning, and Full-Stack Development.

---


