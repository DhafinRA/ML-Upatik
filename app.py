import os
from keras.models import load_model
from flask import Flask, request, jsonify
import pickle

#model_image_path = os.path.join('.', 'foodnutrition.h5')
#model_image = load_model(model_image_path)

#with open('foodnutrition.pkl','rb') as f:
    #model = pickle.load(f)

model = pickle.load(open('foodnutrition.pkl', 'rb'))

flask_app = Flask(__name__)

@app.route('/images', methods=['POST'])
def image_predict():
    data = request.get_json()
    image_url = data['image_url']

    # Assuming preprocessed_image_from_url is defined somewhere
    preprocessed_image = preprocessed_image_from_url(image_url)
    prediction = model_image.predict(preprocessed_image)

    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    flask_app.run(debug=True)
