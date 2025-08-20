# 🌾 Crop Yield Prediction App

A machine learning-based web application that predicts the crop yield (hg/ha) for different countries and crops based on environmental and agricultural features.

---

## 🚀 Features

- Predicts crop yield using a trained **Random Forest Regressor**
- Handles categorical variables with **One-Hot Encoding**
- Applies **standardization** to numerical features
- Built with **Flask** and **Streamlit** for deployment flexibility

---

## 📊 Input Features

- `Year`
- `Average Rainfall (mm/year)`
- `Pesticides Used (tonnes)`
- `Average Temperature (°C)`
- `Country (Area)`
- `Crop Type (Item)`

---

## 🧠 Model Used

- **Random Forest Regressor** (trained with scikit-learn)

---

## 🧪 Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Streamlit
- Flask
- Pickle

---

## 🖥️ How to Run

### 🔧 Clone the Repository

```bash
git clone https://github.com/Yashraj0906/crop-yield-prediction-app.git
cd crop-yield-prediction-app
