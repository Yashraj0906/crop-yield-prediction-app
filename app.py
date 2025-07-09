from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ✅ Load your dataset
df = pd.read_csv("yield.csv")   

# ✅ Preprocessing
col = ['Year', 'average_rain_fall_mm_per_year','pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']
df = df[col]

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ One-hot encode
combined = pd.concat([X_train, X_test])
combined_encoded = pd.get_dummies(combined, columns=['Area', 'Item'], drop_first=True)
X_train_encoded = combined_encoded.iloc[:len(X_train), :]
X_test_encoded = combined_encoded.iloc[len(X_train):, :]

# ✅ Save encoder columns
encoder_columns = X_train_encoded.columns

# ✅ Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

# ✅ Train model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# ✅ Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect input
        Year = float(request.form['Year'])
        rain = float(request.form['average_rain_fall_mm_per_year'])
        pest = float(request.form['pesticides_tonnes'])
        temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        # Prepare input dataframe
        input_df = pd.DataFrame([{
            'Year': Year,
            'average_rain_fall_mm_per_year': rain,
            'pesticides_tonnes': pest,
            'avg_temp': temp,
            'Area': Area,
            'Item': Item
        }])

        # One-hot encode input
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=encoder_columns, fill_value=0)

        # Scale input
        input_scaled = scaler.transform(input_encoded)

        # Predict
        prediction = rf_model.predict(input_scaled)[0]
        return render_template('index.html', prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
