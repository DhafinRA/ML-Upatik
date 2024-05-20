import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn import preprocessing
import pickle
import os
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)
model = pickle.load(open('foodnutrition_baru.pkl', 'rb'))
    
class_names = ['Ayam Bakar','Ayam Geprek',
 'Ayam Goreng','Ayam Tepung','Bakso','Chicken Katsu',
 'Donat','Gado-Gado', 'Kopi', 'Mie Ayam', 'Mie Instan',
'Nasi Goreng','Pecel Lele','Rendang','Sate','Sop','Soto',
 'Telur Balado']

def getCalorie():
    nutrisi_csv = pd.read_csv('nutrisi_origin.csv')
    Kalori = nutrisi_csv['Kalori']
    Protein = nutrisi_csv['Protein']
    Lemak = nutrisi_csv['Lemak']

    return Kalori,Protein,Lemak

@app.route('/images', methods=['POST'])
def image_predict():
    data = request.get_json()
    if 'image_url' not in data:
        return jsonify({'error': 'No image_url provided'}), 400
    
    image_url = data['image_url']
    preprocessed_image = preprocess_image_from_url(image_url)
    if preprocessed_image is None:
        return jsonify({'error': 'Failed to process the image from URL'}), 400

    prediction = model.predict(preprocessed_image)
    return jsonify({'prediction': prediction.tolist()})

def preprocess_image_from_url(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check for HTTP errors
        img = Image.open(BytesIO(response.content))
        img = img.resize((150, 150))  # Adjust the size as needed
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add an extra dimension for batch size
        img_array /= 255.0  # Rescale to the range [0, 1]
        return img_array
    except Exception as e:
        print(f"Error processing image from URL: {e}")
        return None
    
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
