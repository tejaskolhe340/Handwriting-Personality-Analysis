# ✍️ Handwriting-Based Personality Analysis using Machine Learning

This project presents an end-to-end Machine Learning system that analyzes handwriting images and predicts personality traits based on graphology-inspired features. It combines image processing, feature engineering, and supervised learning to build a practical ML application.


## 📌 Problem Statement
Handwriting contains behavioral patterns that can be quantified using image processing techniques.  
The objective of this project is to extract meaningful handwriting features and use them to classify personality traits using Machine Learning.


## 🧠 Approach & Workflow
1. **Image Preprocessing**
   - Noise removal
   - Grayscale conversion
   - Thresholding and contour detection

2. **Feature Extraction**
   - Slant of handwriting  
   - Word and letter spacing  
   - Margins and baseline alignment  
   - Structural handwriting characteristics  

3. **Feature Engineering**
   - Conversion of visual traits into numerical feature vectors
   - Data normalization and preparation for ML models

4. **Model Training**
   - Supervised learning using **Support Vector Machine (SVM)**
   - Hyperparameter tuning and validation

5. **Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report

6. **Deployment**
   - Interactive prediction interface using **Streamlit**


## 🛠️ Technologies Used
- **Python**
- **OpenCV** – image processing
- **NumPy & Pandas** – data handling
- **Scikit-learn** – SVM model & evaluation
- **Streamlit** – web application interface


## 📊 Results
- **Model:** Support Vector Machine (SVM)
- **Accuracy Achieved:** ~80%
- Demonstrated reliable classification of personality traits from handwriting features



