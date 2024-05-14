from fastapi import FastAPI,File,UploadFile
import numpy as np
import PIL.Image
import PIL.ImageOps
import io
import pickle
import os
os.system("cls")
app = FastAPI()

with open('model.pkl','rb') as f:
    model = pickle.load(f)

@app.get('/')
async def hello():
    return "hello"

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img = await file.read()
    img = PIL.Image.open(io.BytesIO(img)).convert('RGB')
    img = img.resize(size=(224,224))
    img = np.array(img)
    img = np.expand_dims(img,axis=0)
    preds = model.predict(img)
    
    # return {"Prediction" : int(preds[0])}
    return preds

hasil = predict()
print(hasil)