from flask import Flask, jsonify, request
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

app = Flask(__name__)

# Download the model when the application starts
model_url = 'https://drive.google.com/uc?id=1ay2Zkhi6e8Y5Z1W8AUP1z-t9UVvAK0P8'
model_path = 'model.h5'  # Choose a local path to save the model file

response = requests.get(model_url)
with open(model_path, 'wb') as f:
    f.write(response.content)

# Load the model
loaded_model = load_model(model_path)

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']

        # Load and preprocess the image
        img = image.load_img(image_file, target_size=(299, 299))  # Assuming InceptionV3 input size
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Perform prediction using the loaded model
        predictions = loaded_model.predict(img_array)

        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
