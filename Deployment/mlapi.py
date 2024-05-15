from fastapi import FastAPI,File,UploadFile
import numpy as np
import PIL.Image
import PIL.ImageOps
import io
import pickle
import os
import pandas as pd
<<<<<<< HEAD
<<<<<<< HEAD
#os.system("cls")
app = FastAPI()

with open('foodnutrition.pkl','rb') as f:
=======
=======
>>>>>>> a900ff1afc5c0ec2e569e1cc957b6c59e455ae2a
os.system("cls")
app = FastAPI()

with open('foodXception.pkl','rb') as f:
<<<<<<< HEAD
>>>>>>> a900ff1afc5c0ec2e569e1cc957b6c59e455ae2a
=======
>>>>>>> a900ff1afc5c0ec2e569e1cc957b6c59e455ae2a
    model = pickle.load(f)

def getCalorie():
    nutrisi_csv = pd.read_csv('nutrisi_origin.csv')
    Kalori = nutrisi_csv['Kalori']
    Protein = nutrisi_csv['Protein']
    Lemak = nutrisi_csv['Lemak']

    return Kalori,Protein,Lemak

@app.get('/')
async def hello():
    return "hello"
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    img = await file.read()
    img = PIL.Image.open(io.BytesIO(img)).convert('RGB')
    img = img.resize(size=(244,244))
    img = np.array(img)
    img = np.expand_dims(img,axis=0)
    preds = model.predict(img)

    Kalori,Protein,Lemak = getCalorie()
  
    # return f"Prediction : {np.argmax(preds[0])}"
<<<<<<< HEAD
<<<<<<< HEAD
    return {"Prediction" : int(np.argmax(preds[0]))}
=======
=======
>>>>>>> a900ff1afc5c0ec2e569e1cc957b6c59e455ae2a
    return {"Prediction" : int(np.argmax(preds[0])),
            "Kalori" : int(Kalori),
            "Protein" : Protein,
            "Lemak" : Lemak}
<<<<<<< HEAD
>>>>>>> a900ff1afc5c0ec2e569e1cc957b6c59e455ae2a
=======
>>>>>>> a900ff1afc5c0ec2e569e1cc957b6c59e455ae2a
# img = PIL.Image.open(io.BytesIO(img)).convert('RGB')
# img = img.resize(size=(224,224))
# img = np.array(img)
# img = np.expand_dims(img,axis=0)
# preds = model.predict(img)
# print(preds)