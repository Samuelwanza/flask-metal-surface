from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import tempfile

app = Flask(__name__)
CORS(app)

# Download the model when the application starts
model_url = 'https://drive.google.com/uc?id=1ay2Zkhi6e8Y5Z1W8AUP1z-t9UVvAK0P8'
model_path = 'model.h5'  # Choose a local path to save the model file
class_labels=['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
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
    print("Predict method called!")

    try:
        # Get the image file from the request
        image_file = request.files['image']
        temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
        image_file.save(temp_file_path)

        # Load and preprocess the images
        img = image.load_img(temp_file_path, target_size=(224, 224))  # Assuming the original input size of InceptionV3
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Perform prediction using the loaded model
        predictions = loaded_model.predict(img_array)

        # Get the index of the maximum value (predicted class)
        predicted_class_index = np.argmax(predictions)

        # Get the corresponding class label
        predicted_class = class_labels[predicted_class_index]

        print("Predicted class index:", predicted_class_index)
        # Return the predicted class as JSON
        return jsonify({'predicted_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=os.getenv("PORT", default=5000))
