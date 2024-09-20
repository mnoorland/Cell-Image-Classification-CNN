# Cell Image Classification Using CNNs and Hand-Crafted Features

## Project Overview
This project involved classifying electron microscope images of cells into eight distinct classes: **basophils**, **eosinophils**, **erythroblasts**, **immature granulocytes**, **lymphocytes**, **neutrophils**, **monocytes**, and **platelets**. We compared the performance of two approaches: traditional **hand-crafted features** and **Convolutional Neural Networks (CNNs)**. The CNN model achieved an accuracy of approximately 90%, significantly outperforming the baseline model built using hand-crafted features.

## Objectives
- **Classify cell images** into eight distinct classes using both CNNs and hand-crafted features.
- **Compare the performance** of traditional feature extraction methods with deep learning approaches.
- **Evaluate model performance** based on accuracy and optimization techniques like hyperparameter tuning.

## Key Features
- **Hand-Crafted Feature Model**: Extracted features such as mean, standard deviation, and GLCM properties (contrast, dissimilarity, homogeneity, energy, correlation) and trained a RandomForestClassifier.
- **CNN Model**: Designed a CNN architecture with three convolutional layers, followed by max-pooling, flattening, dense layers, and dropout for regularization.
- **Experiments**: Conducted hyperparameter tuning, and data augmentation to improve model performance.

## Tools & Technologies
- **TensorFlow2 (Google Colab)**: Used to build and train the CNN model.
- **Python (Jupyter Notebook)**: Used for data preprocessing, feature extraction, and training.
- **Random Forest Classifier**: Applied to the hand-crafted features for comparison with CNNs.
  
## Files Included
- **[Jupyter Notebook](./FinalNotebook.ipynb)**: Contains the code for preprocessing, feature extraction, model training, and evaluation.
- **[Final Report](./Report.pdf)**: A detailed explanation of the project, including data analysis, model development, and results.

## Dataset
The dataset consists of 15,092 electron microscope images, each 48x48 pixels in size, categorized into the following classes:
1. Basophils
2. Eosinophils
3. Erythroblasts
4. Immature Granulocytes
5. Lymphocytes
6. Neutrophils
7. Monocytes
8. Platelets

## Model Performance
- **Hand-Crafted Feature Model**: Achieved an accuracy of approximately 70%.
- **CNN Model**: Achieved an accuracy of approximately 90%.
