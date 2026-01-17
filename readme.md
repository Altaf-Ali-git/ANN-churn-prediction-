# Customer Churn Prediction using ANN

## 📌 Project Overview

This project focuses on predicting whether a customer will **stay with or leave a company** using an **Artificial Neural Network (ANN)** model.  
The solution is divided into three main parts:

1. **Model Training** – Training the ANN on a churn dataset  
2. **Model Prediction** – Using the trained model to predict customer churn  
3. **Streamlit App** – For visualization and deployment of the model

The goal is to help businesses identify customers who are likely to churn and take proactive measures to retain them.

---

## 🧠 Problem Statement

Customer churn is a major challenge for many companies. By analyzing historical customer data, we can predict whether a customer is likely to leave the company. This project uses deep learning (ANN) to solve this binary classification problem.

---

## 🗂 Project Structure

├── data/
│ └── churn_data.csv
├── model/
│ └── ann_model.h5
├── notebooks/
│ └── model_training.ipynb
├── app.py # Streamlit application
├── train.py # Model training script
├── requirements.txt
└── README.md


---

## ⚙️ Tech Stack

- **Programming Language:** Python  
- **Libraries & Frameworks:**
  - NumPy
  - Pandas
  - Scikit-learn
  - TensorFlow / Keras
  - Matplotlib / Seaborn
  - Streamlit
- **IDE/Tools:**
  - VS Code
  - Jupyter Notebook

---

## 🚀 Model Training

- The churn dataset is preprocessed (handling missing values, encoding categorical variables, feature scaling).
- An **Artificial Neural Network (ANN)** is built using Keras.
- The model is trained on the training dataset and validated for accuracy.
- The trained model is saved for later use in predictions.

---

## 🔍 Model Prediction

- The saved ANN model is loaded.
- New customer data is passed as input.
- The model predicts whether the customer will:
  - **Stay (0)**
  - **Leave (1)**

---

## 📊 Streamlit Web App

The Streamlit app provides:
- User-friendly input fields for customer data  
- Real-time churn prediction  
- Clean and interactive UI for easy understanding  

To run the app:



streamlit run app.py

Name: Altaf Ali

Role: Data Science / Machine Learning Enthusiast

Email: altafali086789@gmail.com

GitHub: https://github.com/Altaf-Ali-git

LinkedIn: https://www.linkedin.com/in/altaf-ali-9964b2308/

📜 License

This project is licensed under the MIT License. Feel free to use and modify it.