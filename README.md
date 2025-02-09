# Fetal-Head-Abnormalities-Classification-using-ResNet50-and-MobileNetV3-Small
This project aims to analyze and compare the performance of state-of-the-art deep learning models in classifying fetal head abnormalities using the Fetal Head Abnormalities Dataset available on Kaggle. The primary focus is on implementing ResNet50 and MobileNetV3-Small.

## Problem Statement
Medical imaging plays a crucial role in prenatal healthcare by detecting abnormalities at an early stage, which can lead to timely interventions. The classification of fetal head abnormalities using deep learning can assist radiologists and clinicians in automating the screening process, reducing manual workload, and improving diagnostic accuracy.\
In this assignment, we aim to develop and compare two widely used convolutional neural networks (**ResNet50** and **MobileNetV3-Small**) for the classification of fetal head abnormalities. We explore the impact of different **augmentation techniques** and incorporate **Few-Shot Learning** to assess how well the models perform with limited labeled data.\
## Objectives
**1.Implement and train:**\
• **ResNet50:** A deep residual network architecture known for its strong feature extraction capabilities.\
![image](https://github.com/user-attachments/assets/cda0c2f5-f4ae-468e-b082-fc3bf747e833)\
• **MobileNetV3-Small:** A lightweight and efficient model optimized for mobile and edge devices.\
![image](https://github.com/user-attachments/assets/32a701b9-3972-4e38-9b3a-feccc62d0826)\
**2.Perform model training using:**\
• **Zero-Shot Learning (ZSL)** techniques: The ability of models to generalize without explicitly being trained on unseen abnormalities.\
• **Few-Shot Learning** (FSL): Training models with very few labeled examples per class.\
**3.Investigate the effect of different **data augmentation strategies**:**\
**Traditional augmentations:**\
   - Rotation\
   - Horizontal Flip\
   - Cropping\
## Advanced augmentations:
**Mixup:** A data augmentation technique that blends two images and their labels to improve generalization.\
![image](https://github.com/user-attachments/assets/d4b446d3-403b-49be-85e4-8195929f9a15)\
**CutMix:** Replaces a portion of an image with a patch from another image, encouraging stronger regularization.\
![image](https://github.com/user-attachments/assets/941975b8-e0e2-4e4c-9e58-1907cd83788c)\
**4.Evaluate performance using standard classification metrics:**\
  - Confusion Matrix\
  - Accuracy\
  - Precision\
  - Recall\
  - F1-Score\\
**5.Compare and analyze** the results obtained from the two models.
## Dataset Details :
The dataset used in this project is the **Fetal Head Abnormalities Dataset**, which consists of ultrasound images categorized into different classes of abnormalities. Proper preprocessing is required before training the models.\
### Key challenges of the dataset:
•**Class Imbalance:** Some fetal head abnormalities might have significantly fewer samples than others.\
•**High Intra-Class Variability:** Variations in image quality, angles, and noise.\
•**Small Sample Size:** Few-Shot Learning techniques must be leveraged to improve generalization.

## Methodology
This project follows a structured deep learning pipeline to systematically preprocess data, train models, evaluate performance, and compare results.

### 1. Data Preprocessing
Image resizing and normalization.  
Augmenting images using both traditional and advanced augmentation techniques.  
Splitting the dataset into training, validation, and test sets.  
Handling class imbalance using techniques such as weighted loss functions or oversampling/undersampling.  

### 2. Model Implementation
Both **ResNet50** and **MobileNetV3-Small** are implemented using PyTorch and trained from scratch or using transfer learning. Their architectures are:  

**ResNet50**: A deep residual network with skip connections that mitigate vanishing gradients.  
**MobileNetV3-Small**: A lightweight CNN optimized for mobile applications, using depthwise separable convolutions.  

### 3. Training Strategies
Three different training strategies are employed:  

#### - Traditional Augmentations (Zero-Shot Learning)
  - Images are augmented using rotation, flipping, and cropping.  

#### - Advanced Augmentations (Zero-Shot Learning)
  - Mixup and CutMix are used to improve generalization.  

#### - Few-Shot Learning (FSL)
  - Models are trained with very few labeled examples to evaluate their ability to learn effectively with limited data.  

### 4. Evaluation Metrics
Model performance is assessed using:  

**Confusion Matrix**: Visual representation of true vs. predicted labels.  
**Accuracy**: Overall correctness of predictions.  
**Precision**: The proportion of true positive classifications out of all predicted positives.  
**Recall**: The proportion of true positive classifications out of all actual positives.  
**F1-Score**: Harmonic mean of precision and recall.  

### 5. Comparison and Analysis
After training and evaluation, both models are compared based on:  

Performance on different augmentation techniques.  
Impact of Few-Shot Learning on classification accuracy.  
Trade-offs between accuracy and computational efficiency.  
Real-world feasibility for medical applications.

## Technologies Used
**Deep Learning Framework**: PyTorch  
**Pretrained Models**: ResNet50, MobileNetV3-Small  
**Data Augmentation Techniques**: Albumentations, Torchvision Transforms  
**Evaluation & Visualization**: Matplotlib, Seaborn, Scikit-learn  
**Few-Shot Learning**: Prototypical Networks or Transfer Learning  

## Expected Outcomes
Comparison of **ResNet50** and **MobileNetV3-Small** for medical image classification.  
Insights on the effectiveness of **traditional vs. advanced augmentation techniques**.  
Impact of **Few-Shot Learning** in handling medical datasets with limited annotations.  
Best-performing model for **fetal head abnormality classification**.  

## Conclusion
This project serves as an extensive comparative study on the application of **deep learning for medical image classification**. It highlights the **advantages and challenges** of using ResNet50 and MobileNetV3-Small under different **augmentation techniques** and **few-shot learning scenarios**. The findings from this research can contribute to developing **automated, efficient, and scalable solutions for medical diagnostics in prenatal care**.  

### Future Work
For future work, the approach can be extended to:  

Explore additional deep learning architectures such as **EfficientNet** and **Vision Transformers (ViTs)**.  
Incorporate **semi-supervised or self-supervised learning techniques**.  
Develop a **real-time deployment framework for clinical use**.
