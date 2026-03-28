E-Waste Image Classification Using EfficientNetV2B0 (Transfer Learning)
Problem Statement and Description
E-waste (electronic waste) is rapidly becoming a serious environmental and health issue around the world. Proper sorting and categorization of e-waste is essential for efficient recycling and disposal, but manual classification is error-prone and labor-intensive.

This project aims to build an automated e-waste classification system using artificial intelligence and machine learning. By training a deep learning model on images of different types of e-waste, we can identify and categorize them accurately.

Goal:
Use image classification with EfficientNetV2B0 to classify e-waste into 10 distinct categories to support better sorting and recycling automation.

Dataset Overview
Dataset Name: E-Waste Image Dataset
Source: https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset
Each directory contains 10 subfolders, each representing one class of e-waste:

PCB (Printed Circuit Board)
Player
Battery
Microwave
Mobile
Mouse
Printer
Television
Washing Machine
Keyboard

What is Transfer Learning?
Transfer Learning: Transfer Learning is a machine learning technique where a pre-trained model developed for a specific task is reused as the starting point for a model on a different but related task. It also allows us to build accurate models in a time-saving way by starting from patterns learned when solving a different problem. This approach is beneficial when there is limited data for the new task, as the pre-trained model already has learned features that can be adapted. Transfer learning can significantly improve models' performance and efficiency in domains like computer vision and natural language processing.

Benefits
Reduces training time — you don't start from scratch.
Leverages learned features from large datasets (like ImageNet).
Improves performance, especially with limited data.
How Does It Work?
Load a pretrained model (e.g., ResNet, EfficientNet).
Freeze the pretrained layers (optional).
Add new layers for your custom task.
Train on your new dataset (can also fine-tune).
EfficientNetV2B0: Transfer Learning Backbone
Overview
EfficientNetV2 is an optimized family of models introduced by Google for efficient training and inference.

Key Features:
Fused MBConv blocks — improve training speed and GPU efficiency.
Progressive learning — gradually increases input size during training.
Better accuracy with fewer parameters and FLOPs.
Why Use EfficientNetV2B0?
Lightweight - Small model size, ideal for mobile & edge devices
Fast - Quick training and inference
Pretrained on ImageNet - Excellent feature extraction baseline
High Accuracy - Competitively performs even in low-resource setups

Core Libraries
tensorflow: For deep learning model building and training.
numpy: For numerical operations and array manipulation.
matplotlib.pyplot: For plotting training curves and results.

Format: Folder-based image classification dataset
Train/: Images used for training the model
Test/: Images used for model evaluation
Validation/: Images used to fine-tune and validate the model
