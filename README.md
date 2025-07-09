# 🌾 Crop Yield Prediction App

A machine learning-based web application that predicts the crop yield (hg/ha) for different countries and crops based on environmental and agricultural features.

## 🚀 Features

- Predicts crop yield using a trained Random Forest model
- Handles categorical variables using One-Hot Encoding
- Applies standardization for numerical features
- Built with Streamlit for an interactive UI

## 📊 Input Features

- Year
- Average Rainfall (mm/year)
- Pesticides Used (tonnes)
- Average Temperature (°C)
- Country (Area)
- Crop Type (Item)

## 🧠 Model Used

- **Random Forest Regressor**

## 🧪 Technologies Used

- Python
- Pandas, NumPy
- scikit-learn
- Streamlit
- Pickle (for model and scaler saving)

## 🖥️ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Yashraj0906/crop-yield-prediction-app.git
   cd crop-yield-prediction-app
