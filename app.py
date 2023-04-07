

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from ImageNotes import ImageNote
import numpy as np
import pickle
import pandas as pd
from main import getPrediction


# 2. Create the app object
app = FastAPI()


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To KeVan Youtube Channel': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_crack(data:ImageNote):
    data = data.dict()
    img_url=data['image_url']

    pred= getPrediction(img_url)
    print(pred)
    # if(prediction[0][0]>0.5):
    #     prediction="Have crack"
    # else:
    #     prediction="Have no crack"
    # return {
    #     'prediction': str(pred[0][0])
    # }

    return {
        'prediction': str(pred)
    }


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
