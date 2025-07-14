from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ✅ Load model from same directory as app.py
model_path = os.path.join(os.path.dirname(__file__), 'house_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    area = float(request.form['area'])
    quality = int(request.form['quality'])
    year = int(request.form['year'])

    features = np.array([[area, quality, year]])
    prediction = model.predict(features)[0]

    return render_template('index.html', result=round(prediction, 2))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
