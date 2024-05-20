import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle
import os

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model = pickle.load(open('foodnutrition_baru.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/image_predict', methods=['POST'])
def image_predict():
    data = request.get_json()
    image_url = data['image_url']

    # Assuming preprocessed_image_from_url is defined somewhere
    preprocessed_image = preprocessed_image_from_url(image_url)
    prediction = model.predict(preprocessed_image)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
